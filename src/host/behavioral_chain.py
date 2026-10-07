"""Behavioral chain harness — state + decision links (VALIDATION §7.1).

The behavioral auto-test checks the host's **state channel** link by link
(VALIDATION §7). This module covers the first two links:

* **Link 1 — state**: the affective state is derived from the right input,
  bounded, and not NaN/inf. Checked by *state invariants* (VALIDATION §2),
  never by an "expected value" — state is a *derived* quantity, not a
  measurement (VALIDATION §7.2).
* **Link 2 — decision**: speaking/silent/initiative is an *observable* choice
  with a causal trace (``PolicyTrace``). Checked by the *reaction class*
  (VALIDATION §7.3): silence is a decision too, not an absence.
* **Link 3 — intent**: the ``IntentFrame`` must be *grounded* in the state and
  the decision — the goal follows from the policy action (full ``Action → goal``
  mapping), the affect/numbers derive from the same state, the task matches.
  Checked by *intent invariants* (VALIDATION §7.4): the LLM sees the frame, not
  the state, so intent distortion is caught here separately from the reply.

Functional Core (pure, ADR-0004):

* :class:`ReactionClass` — the observable decision class.
* :class:`StateInvariant` — a boundedness/finiteness property of the state.
* :class:`StateView` — a compact snapshot of the state (link 1).
* :func:`classify_reaction` — ``PolicyTrace`` (+ escape hatch) → class.
* :func:`check_state` — which invariants a snapshot violates.
* :class:`IntentInvariant` — a grounding property of the intent (link 3).
* :class:`IntentView` — the frame plus the state it was derived from.
* :func:`check_intent` / :func:`intent_from_state` — intent grounding checks.
* :class:`Scenario` / :class:`ScenarioResult` — one test cell and its outcome.
* :func:`default_scenarios` — the built-in corpus (links 1–2).
* :class:`Ablation` / :class:`AblationCheck` / :class:`AblationResult` —
  attribution checks: disabling a mechanism must *change* the observable
  (VALIDATION §7.5). Non-tautological: if nothing changes, the test was about
  substitution, not the mechanism.
* :func:`default_ablations` — the built-in attribution corpus.

Imperative Shell:

* :class:`BehavioralChainRunner` — runs a scenario over ``HostLoop`` with
  deterministic knobs (preset + synthetic clock + fake embedder + fake meter),
  driving policy itself (no LLM, ADR-0007), mirroring ``ChatSession``'s decision
  path. Same seed → same result (VALIDATION §7.7). ``run_ablation`` runs the
  same scenario twice (mechanism on/off) and compares the observable.

Not covered yet (by design): ToM (``partner`` is ``None`` → S4-compat),
actuation/reply (links 4–5 — LLM), and preconditions ``primed``/``matured``
(only ``born`` is implemented; VALIDATION §7.8).
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import HostConfig, load_preset
from src.core.policy import Action, PolicyTrace, select_action
from src.host.fingerprint import FINGERPRINT_METRICS, behavioral_fingerprint
from src.host.loop import HostLoop, build_host_loop
from src.host.sensitivity import DeterministicMeter
from src.speech.intent import (
    GOAL_INSTRUCTIONS,
    IntentFrame,
    build_intent_frame,
    describe_affect,
    goal_for_action,
)

_SPEAKING = frozenset(
    {Action.RESPOND, Action.INITIATIVE, Action.IDENTIFY_PARTNER}
)


class ReactionClass(Enum):
    """Наблюдаемый класс решения (звено «решение», VALIDATION §7.1).

    Молчание (``SILENT``) — такой же наблюдаемый класс, как и речь
    (VALIDATION §7.3), а не отсутствие реакции.
    """

    RESPOND = "respond"
    SILENT = "silent"
    INITIATIVE = "initiative"
    IDENTIFY_PARTNER = "identify_partner"
    EXPLORE = "explore"
    ESCAPE_HATCH = "escape_hatch"


_ACTION_TO_REACTION: dict[Action, ReactionClass] = {
    Action.RESPOND: ReactionClass.RESPOND,
    Action.SILENT: ReactionClass.SILENT,
    Action.INITIATIVE: ReactionClass.INITIATIVE,
    Action.IDENTIFY_PARTNER: ReactionClass.IDENTIFY_PARTNER,
    Action.EXPLORE: ReactionClass.EXPLORE,
}


class StateInvariant(Enum):
    """Наблюдаемое свойство состояния (звено «состояние», VALIDATION §7.2)."""

    FINITE = "finite"
    F_NONNEG = "f_nonneg"
    STRESS_NONNEG = "stress_nonneg"
    PARTNER_BOUNDED = "partner_bounded"


class IntentInvariant(Enum):
    """Наблюдаемое свойство интента (звено «интент», VALIDATION §7.4).

    Интент — это **представление состояния для LLM**, а не состояние. Поэтому
    проверяется *grounding* (соответствие состояния frame'у), а не «правильный
    текст»:
        GOAL_CONSISTENT — цель frame'а согласована с решением policy (маппинг
            ``Action → goal`` соблюдён, ``EXPLORE`` не выдаётся за ``respond``);
        AFFECT_CONSISTENT — ``affect``/числа frame'а выведены из того же
            состояния (valence/stress не подменены);
        TASK_CONSISTENT — ``task`` frame'а совпадает с активной задачей хоста.
    """

    GOAL_CONSISTENT = "goal_consistent"
    AFFECT_CONSISTENT = "affect_consistent"
    TASK_CONSISTENT = "task_consistent"


_DEFAULT_INTENT_INVARIANTS: tuple[IntentInvariant, ...] = (
    IntentInvariant.GOAL_CONSISTENT,
    IntentInvariant.AFFECT_CONSISTENT,
    IntentInvariant.TASK_CONSISTENT,
)


class Ablation(Enum):
    """Выключаемый механизм для проверки атрибуции (VALIDATION §7.5).

    Ablation не тавтологичен: проверяется, что наблюдаемое свойство
    **меняется** при выключении механизма. Если не меняется — тест был про
    подмену, а не про механизм.
    """

    POLICY = "policy"  # policy.enabled=False
    MEMORY = "memory"  # memory.enabled=False (канал + провайдер)
    RECALL = "recall"  # memory.recall_enabled=False (только содержимое)


def _ablate(config: HostConfig, ablation: Ablation) -> HostConfig:
    """Вернуть конфиг с выключенным механизмом (чистая).

    ``RECALL`` выключает **только содержимое воспоминаний** (prior «молчит»),
    сохраняя провайдер сообщения и размерность шины — это чистый ablation
    содержания памяти. ``MEMORY`` выключает память целиком (канал + провайдер),
    поэтому он грубее: подмена коммуникативного входа делает различие F
    неоднозначным (confounding).

    Args:
        config: Базовый конфиг.
        ablation: Выключаемый механизм.

    Returns:
        Новый HostConfig с выключенным механизмом.
    """
    if ablation is Ablation.POLICY:
        return replace(config, policy=replace(config.policy, enabled=False))
    if ablation is Ablation.RECALL:
        return replace(config, memory=replace(config.memory, recall_enabled=False))
    return replace(config, memory=replace(config.memory, enabled=False))


@dataclass(frozen=True)
class StateView:
    """Компактный снимок состояния хоста (звено 1).

    Attributes:
        tick: Номер тика.
        f: Свободная энергия F(t).
        valence: Валентность.
        stress: Аллостатический стресс.
        gamma: Точность γ.
        task: Активная задача/аттрактор.
        active_columns: Число активных колонок.
        drift: Флаг дрейфа.
        partner_trust: Доверие к партнёру, [0, 1] (0 без ToM).
        partner_uncertainty: Неопределённость идентичности, [0, 1].
    """

    tick: int
    f: float
    valence: float
    stress: float
    gamma: float
    task: str
    active_columns: int
    drift: bool
    partner_trust: float
    partner_uncertainty: float


def classify_reaction(trace: PolicyTrace, *, escape_hatch: bool = False) -> ReactionClass:
    """Классифицировать решение в наблюдаемый класс (чистая).

    Escape hatch (право голоса под удержанным throttle) имеет приоритет над
    решением policy: он тоже наблюдаемый класс (VALIDATION §7.3).

    Args:
        trace: Причинная трасса решения policy.
        escape_hatch: Активен ли escape hatch в этом тике.

    Returns:
        ReactionClass.
    """
    if escape_hatch:
        return ReactionClass.ESCAPE_HATCH
    return _ACTION_TO_REACTION[trace.chosen]


def check_state(
    view: StateView, invariants: Sequence[StateInvariant]
) -> tuple[str, ...]:
    """Проверить инварианты состояния; вернуть имена нарушенных (чистая).

    Args:
        view: Снимок состояния.
        invariants: Проверяемые инварианты.

    Returns:
        Кортеж имён нарушенных инвариантов (пусто → всё выполнено).
    """
    violated: list[str] = []
    finite = all(
        math.isfinite(v) for v in (view.f, view.valence, view.stress, view.gamma)
    )
    for invariant in invariants:
        if invariant is StateInvariant.FINITE:
            if not finite:
                violated.append(invariant.value)
        elif invariant is StateInvariant.F_NONNEG:
            if not (math.isfinite(view.f) and view.f >= 0.0):
                violated.append(invariant.value)
        elif invariant is StateInvariant.STRESS_NONNEG:
            if not (math.isfinite(view.stress) and view.stress >= 0.0):
                violated.append(invariant.value)
        elif invariant is StateInvariant.PARTNER_BOUNDED and not (
            0.0 <= view.partner_trust <= 1.0
            and 0.0 <= view.partner_uncertainty <= 1.0
        ):
            violated.append(invariant.value)
    return tuple(violated)


_DEFAULT_INVARIANTS: tuple[StateInvariant, ...] = (
    StateInvariant.FINITE,
    StateInvariant.F_NONNEG,
    StateInvariant.STRESS_NONNEG,
)


@dataclass(frozen=True)
class IntentView:
    """Снимок звена 3: frame + состояние, из которого он выведен.

    Интент проверяется на *grounding*: frame должен быть выведен из текущего
    состояния и решения, а не подменён (VALIDATION §7.4).

    Attributes:
        frame: Собранный ``IntentFrame``.
        action: Решение policy, из которого выведена цель (None → S3-фолбэк).
        state_valence: Валентность состояния-источника.
        state_stress: Стресс состояния-источника.
        state_task: Активная задача состояния-источника.
    """

    frame: IntentFrame
    action: Action | None
    state_valence: float
    state_stress: float
    state_task: str


def check_intent(
    view: IntentView, invariants: Sequence[IntentInvariant]
) -> tuple[str, ...]:
    """Проверить grounding интента; вернуть имена нарушенных (чистая).

    Args:
        view: Снимок интента (frame + состояние-источник).
        invariants: Проверяемые инварианты.

    Returns:
        Кортеж имён нарушенных инвариантов (пусто → всё выполнено).
    """
    violated: list[str] = []
    for invariant in invariants:
        if invariant is IntentInvariant.GOAL_CONSISTENT:
            ok = view.frame.goal in GOAL_INSTRUCTIONS and (
                view.action is None
                or goal_for_action(view.action) == view.frame.goal
            )
        elif invariant is IntentInvariant.AFFECT_CONSISTENT:
            finite = all(
                math.isfinite(v)
                for v in (
                    view.frame.valence,
                    view.frame.stress,
                    view.frame.free_energy,
                )
            )
            ok = (
                finite
                and view.frame.affect
                == describe_affect(view.state_valence, view.state_stress)
                and view.frame.valence == view.state_valence
            )
        elif invariant is IntentInvariant.TASK_CONSISTENT:
            ok = view.frame.task == view.state_task
        else:  # pragma: no cover — все члены перечислены
            ok = True
        if not ok:
            violated.append(invariant.value)
    return tuple(violated)


def intent_from_state(view: StateView, action: Action | None) -> IntentView:
    """Собрать интент из состояния и решения policy (чистая, звено 3).

    Повторяет путь ``ChatSession``: цель выводится из решения policy полным
    маппингом (``goal_for_action``), frame — из текущего состояния. ``None``
    → S3-фолбэк (``respond``).

    Args:
        view: Снимок состояния (звено 1).
        action: Решение policy или None (S3-совместимость).

    Returns:
        ``IntentView`` с frame'ом и состоянием-источником.
    """
    goal = goal_for_action(action) if action is not None else "respond"
    frame = build_intent_frame(
        f=view.f,
        valence=view.valence,
        stress=view.stress,
        task=view.task,
        goal=goal,
    )
    return IntentView(
        frame=frame,
        action=action,
        state_valence=view.valence,
        state_stress=view.stress,
        state_task=view.task,
    )


@dataclass(frozen=True)
class Scenario:
    """Одна ячейка поведенческого теста (сценарий → ожидаемый класс).

    Attributes:
        id: Идентификатор (например ``state.bounds.baseline``).
        link: Проверяемое звено (``state`` | ``decision``).
        preset: Имя пресета (ручки заморожены).
        seed: Зерно детерминированных провайдеров.
        ticks: Число тиков прогона.
        messages: Скрипт сообщений ``(tick, text)``.
        expect_reaction: Ожидаемый класс реакции (None → не проверяется).
        state_invariants: Проверяемые инварианты состояния.
        precondition: Предусловие (``born``; ``primed``/``matured`` — позже).

    Raises:
        ValueError: Если id/звено пусты, ticks < 1 или предусловие неизвестно.
    """

    id: str
    link: str
    preset: str
    seed: int = 0
    ticks: int = 120
    messages: tuple[tuple[int, str], ...] = ()
    expect_reaction: ReactionClass | None = None
    state_invariants: tuple[StateInvariant, ...] = _DEFAULT_INVARIANTS
    intent_invariants: tuple[IntentInvariant, ...] = _DEFAULT_INTENT_INVARIANTS
    precondition: str = "born"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("scenario id must not be empty")
        if self.link not in ("state", "decision", "intent"):
            raise ValueError(
                f"link must be 'state'|'decision'|'intent', got {self.link!r}"
            )
        if self.ticks < 1:
            raise ValueError(f"ticks must be >= 1, got {self.ticks}")
        if self.precondition != "born":
            raise ValueError(
                f"precondition {self.precondition!r} not implemented "
                "(only 'born'; VALIDATION §7.8)"
            )


@dataclass(frozen=True)
class ScenarioResult:
    """Итог прогона сценария.

    Attributes:
        scenario: Прогнанный сценарий.
        reactions: Наблюдаемые классы реакций по тикам (в порядке тиков).
        violations: Имена нарушенных инвариантов состояния (уникальные).
        intent_violations: Имена нарушенных инвариантов интента (уникальные).
        passed: Выполнен ли сценарий (нет нарушений + ожидание совпало).
        reason: Причина вердикта (для диагностики).
    """

    scenario: Scenario
    reactions: tuple[ReactionClass, ...]
    violations: tuple[str, ...]
    intent_violations: tuple[str, ...]
    passed: bool
    reason: str


# Наблюдаемые для ablation: метрика отпечатка, доля policy-решений или
# множество классов реакций. Проверяем изменение, а не значение (VALIDATION §7.5).
_OBSERVABLE_POLICY_RATE = "policy_rate"
_OBSERVABLE_REACTIONS = "reaction_classes"


@dataclass(frozen=True)
class AblationCheck:
    """Проверка атрибуции: выключить механизм → наблюдаемое меняется.

    Абсолютные значения не проверяются — только факт изменения (VALIDATION
    §7.5). Если при выключенном механизме наблюдаемое не меняется, тест
    проверял подмену, а не механизм.

    Attributes:
        id: Идентификатор (например ``ablate.policy.decision``).
        preset: Имя пресета.
        mechanism: Выключаемый механизм.
        observable: Что наблюдаем: метрика отпечатка (``f_mean``, ...),
            ``policy_rate`` или ``reaction_classes``.
        expect_change: Должно ли наблюдаемое измениться (True) или остаться
            прежним (False).
        seed: Зерно.
        ticks: Число тиков.
        messages: Скрипт сообщений ``(tick, text)``.
        tol: Допуск на изменение числовой наблюдаемой.

    Raises:
        ValueError: Если id/наблюдаемая пусты, ticks < 1 или неизвестная
            наблюдаемая.
    """

    id: str
    preset: str
    mechanism: Ablation
    observable: str
    expect_change: bool = True
    seed: int = 0
    ticks: int = 120
    messages: tuple[tuple[int, str], ...] = ()
    tol: float = 1e-9

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ablation id must not be empty")
        if self.ticks < 1:
            raise ValueError(f"ticks must be >= 1, got {self.ticks}")
        if self.observable not in (
            _OBSERVABLE_POLICY_RATE,
            _OBSERVABLE_REACTIONS,
            *FINGERPRINT_METRICS,
        ):
            raise ValueError(f"unknown ablation observable {self.observable!r}")

    def as_scenario(self) -> Scenario:
        """Сценарий-носитель для прогона (без ожидаемого класса)."""
        return Scenario(
            id=self.id,
            link="decision" if self.observable == _OBSERVABLE_REACTIONS else "state",
            preset=self.preset,
            seed=self.seed,
            ticks=self.ticks,
            messages=self.messages,
        )


@dataclass(frozen=True)
class AblationResult:
    """Итог проверки атрибуции.

    Attributes:
        check: Проверка.
        baseline: Наблюдаемое при включённом механизме.
        ablated: Наблюдаемое при выключенном механизме.
        changed: Изменилось ли наблюдаемое.
        passed: Выполнено ли ожидание (change/no-change).
        reason: Причина вердикта.
    """

    check: AblationCheck
    baseline: object
    ablated: object
    changed: bool
    passed: bool
    reason: str


def default_scenarios() -> tuple[Scenario, ...]:
    """Встроенный корпус сценариев звеньев 1–3 (VALIDATION §7.1).

    Returns:
        Кортеж :class:`Scenario`. Сценарии, бессмысленные без истории, на
        ``born`` не входят (VALIDATION §7.8).
    """
    return (
        Scenario(
            id="state.bounds.baseline",
            link="state",
            preset="baseline",
            ticks=120,
        ),
        Scenario(
            id="state.bounds.dialogue",
            link="state",
            preset="dialogue",
            ticks=120,
            messages=((0, "привет"),),
        ),
        Scenario(
            id="decision.respond",
            link="decision",
            preset="dialogue",
            ticks=120,
            messages=((0, "привет"),),
            expect_reaction=ReactionClass.RESPOND,
        ),
        Scenario(
            id="intent.respond.grounding",
            link="intent",
            preset="dialogue",
            ticks=120,
            messages=((0, "привет"),),
        ),
        Scenario(
            id="intent.silent.grounding",
            link="intent",
            preset="dialogue",
            ticks=120,
        ),
    )


def default_ablations() -> tuple[AblationCheck, ...]:
    """Встроенный корпус проверок атрибуции (VALIDATION §7.5).

    Каждая проверка подтверждает, что наблюдаемое **меняется** при выключении
    механизма — иначе тест был бы про подмену, а не про механизм.

    Returns:
        Кортеж :class:`AblationCheck`.
    """
    return (
        # policy выключена → классы решений исчезают (реакций нет).
        AblationCheck(
            id="ablate.policy.decision",
            preset="dialogue",
            mechanism=Ablation.POLICY,
            observable=_OBSERVABLE_REACTIONS,
            messages=((0, "привет"),),
        ),
        # policy выключена → доля policy-решений падает до нуля.
        AblationCheck(
            id="ablate.policy.rate",
            preset="dialogue",
            mechanism=Ablation.POLICY,
            observable=_OBSERVABLE_POLICY_RATE,
            messages=((0, "привет"),),
        ),
        # память выключена целиком (канал + провайдер) → F меняется, но
        # различие смешано с подменой коммуникативного входа (confounding).
        AblationCheck(
            id="ablate.memory.channel",
            preset="dialogue",
            mechanism=Ablation.MEMORY,
            observable="f_mean",
            messages=((0, "привет"),),
        ),
        # Чистый ablation содержания памяти: recall выключен (prior «молчит»),
        # провайдер и размерность сохранены. Сценарий с реальным recall
        # (2 сообщения → memory_hit > 0): воспоминание должно менять F.
        AblationCheck(
            id="ablate.recall.content",
            preset="dialogue",
            mechanism=Ablation.RECALL,
            observable="f_mean",
            messages=((0, "привет"), (60, "снова я")),
        ),
    )


class BehavioralChainRunner:
    """Shell: прогон сценариев поверх ``HostLoop``, сбор классов реакций.

    Attributes:
        workdir: Каталог для логов/БД (детерминизм не зависит от пути).
    """

    def __init__(self, *, workdir: Path | None = None) -> None:
        self.workdir = (
            Path(workdir)
            if workdir is not None
            else Path(tempfile.mkdtemp(prefix="behavioral-chain-"))
        )
        self._counter = 0

    def run(self, scenario: Scenario) -> ScenarioResult:
        """Прогнать один сценарий и вернуть результат.

        Args:
            scenario: Сценарий.

        Returns:
            ScenarioResult с наблюдаемыми реакциями и вердиктом.
        """
        config = self._make_config(scenario)
        reactions, violations, intent_violations, _, _ = self._run_config(
            config, scenario
        )
        unique_violations = tuple(dict.fromkeys(violations))
        unique_intent_violations = tuple(dict.fromkeys(intent_violations))
        expected_ok = (
            scenario.expect_reaction is None
            or scenario.expect_reaction in reactions
        )
        passed = (
            not unique_violations
            and not unique_intent_violations
            and expected_ok
        )
        reason = self._reason(
            scenario,
            reactions,
            unique_violations,
            unique_intent_violations,
            expected_ok,
        )
        return ScenarioResult(
            scenario=scenario,
            reactions=tuple(reactions),
            violations=unique_violations,
            intent_violations=unique_intent_violations,
            passed=passed,
            reason=reason,
        )

    def run_all(
        self, scenarios: Sequence[Scenario] | None = None
    ) -> list[ScenarioResult]:
        """Прогнать корпус сценариев (по умолчанию — встроенный).

        Args:
            scenarios: Сценарии (None → :func:`default_scenarios`).

        Returns:
            Список :class:`ScenarioResult` в порядке сценариев.
        """
        corpus = default_scenarios() if scenarios is None else scenarios
        return [self.run(scenario) for scenario in corpus]

    def run_ablation(self, check: AblationCheck) -> AblationResult:
        """Проверить атрибуцию: выключить механизм → наблюдаемое меняется.

        Прогоняет один и тот же сценарий дважды — с механизмом и без — и
        сравнивает наблюдаемое (VALIDATION §7.5). Абсолютные значения не
        оцениваются, только факт изменения.

        Args:
            check: Проверка атрибуции.

        Returns:
            AblationResult с обоими наблюдаемыми и вердиктом.
        """
        scenario = check.as_scenario()
        base_config = self._make_config(scenario, tag="base")
        ablated_config = _ablate(base_config, check.mechanism)
        # Отдельные каталоги прогонов: телеметрия не должна смешиваться
        # (иначе ablation сравнивал бы прогон сам с собой).
        ablated_config = replace(
            ablated_config, log_path=str(self._run_dir(scenario, "ablated") / "run.jsonl")
        )

        _, _, _, base_policy, base_events = self._run_config(base_config, scenario)
        _, _, _, abl_policy, abl_events = self._run_config(ablated_config, scenario)

        baseline = _observable(check, base_policy, base_events)
        ablated = _observable(check, abl_policy, abl_events)
        changed = _observable_changed(baseline, ablated, tol=check.tol)
        passed = changed == check.expect_change
        reason = _ablation_reason(check, baseline, ablated, changed)
        return AblationResult(
            check=check,
            baseline=baseline,
            ablated=ablated,
            changed=changed,
            passed=passed,
            reason=reason,
        )

    def run_ablations(
        self, checks: Sequence[AblationCheck] | None = None
    ) -> list[AblationResult]:
        """Прогнать корпус проверок атрибуции (по умолчанию — встроенный).

        Args:
            checks: Проверки (None → :func:`default_ablations`).

        Returns:
            Список :class:`AblationResult` в порядке проверок.
        """
        corpus = default_ablations() if checks is None else checks
        return [self.run_ablation(check) for check in corpus]

    def _make_config(self, scenario: Scenario, *, tag: str = "") -> HostConfig:
        """Собрать детерминированный конфиг под сценарий (пресет замораживает)."""
        run_dir = self._run_dir(scenario, tag)
        config = load_preset(scenario.preset)
        return replace(
            config,
            seed=scenario.seed,
            max_ticks=scenario.ticks,
            log_path=str(run_dir / "run.jsonl"),
            memory=replace(config.memory, db_path=str(run_dir / "mem.db")),
        )

    def _run_dir(self, scenario: Scenario, tag: str) -> Path:
        """Выделить отдельный каталог прогона (телеметрия не смешивается)."""
        self._counter += 1
        suffix = f"-{tag}" if tag else ""
        run_dir = self.workdir / f"{self._counter:03d}-{scenario.id}{suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _run_config(
        self, config: HostConfig, scenario: Scenario
    ) -> tuple[list[ReactionClass], list[str], list[str], int, list[dict[str, Any]]]:
        """Прогнать loop под готовым конфигом и собрать наблюдаемые.

        Returns:
            (reactions, violations, intent_violations, policy_events, events):
            классы реакций, нарушенные инварианты состояния, нарушенные
            инварианты интента, число policy-решений и строки телеметрии.
        """
        loop = build_host_loop(
            config, meter=DeterministicMeter(), messages=scenario.messages
        )
        try:
            reactions, violations, intent_violations, policy_events = self._drive(
                loop, config, scenario
            )
            events = _read_events(Path(config.log_path))
        finally:
            loop.close()
        return reactions, violations, intent_violations, policy_events, events

    @staticmethod
    def _drive(
        loop: HostLoop, config: HostConfig, scenario: Scenario
    ) -> tuple[list[ReactionClass], list[str], list[str], int]:
        """Прогнать тики, ведя policy вручную (как ``ChatSession``).

        Новое сообщение на тике ``t`` → решение с ``has_new_message=True``;
        иначе — с ``False`` (возможны инициатива/исследование). LLM не
        вызывается (ADR-0007). При ``policy.enabled=False`` решения не
        принимаются (ablation): классы реакций и счётчик пусты. На каждом тике
        интент собирается из состояния и решения (звено 3) и проверяется на
        grounding.

        Returns:
            (reactions, violations, intent_violations, policy_events).
        """
        message_ticks = {tick for tick, _ in scenario.messages}
        reactions: list[ReactionClass] = []
        violations: list[str] = []
        intent_violations: list[str] = []
        policy_events = 0
        for tick in range(scenario.ticks):
            loop.step_once(tick)
            action: Action | None = None
            if config.policy.enabled:
                has_new_message = tick in message_ticks
                context = loop.policy_context(
                    has_new_message=has_new_message, mode=config.policy.mode
                )
                trace = select_action(context, config.policy.preferences)
                loop.record_policy(trace)
                action = trace.chosen
                if trace.chosen in _SPEAKING:
                    loop.mark_spoke()
                reactions.append(
                    classify_reaction(trace, escape_hatch=loop.escape_hatch_active)
                )
                policy_events += 1
            view = _take_view(loop)
            violations.extend(check_state(view, scenario.state_invariants))
            intent_view = intent_from_state(view, action)
            intent_violations.extend(
                check_intent(intent_view, scenario.intent_invariants)
            )
        return reactions, violations, intent_violations, policy_events

    @staticmethod
    def _reason(
        scenario: Scenario,
        reactions: Sequence[ReactionClass],
        violations: tuple[str, ...],
        intent_violations: tuple[str, ...],
        expected_ok: bool,
    ) -> str:
        """Собрать причину вердикта (для журнала/диагностики)."""
        if violations:
            return f"state invariant violated: {', '.join(violations)}"
        if intent_violations:
            return f"intent invariant violated: {', '.join(intent_violations)}"
        if not expected_ok:
            observed = sorted({r.value for r in reactions})
            expected = scenario.expect_reaction
            expected_value = expected.value if expected is not None else "none"
            return (
                f"expected reaction {expected_value!r} "
                f"not observed (observed: {', '.join(observed) or 'none'})"
            )
        return "ok"


def _observable(
    check: AblationCheck, policy_events: int, events: Sequence[Mapping[str, Any]]
) -> object:
    """Извлечь наблюдаемую проверки (чистая).

    Args:
        check: Проверка атрибуции.
        policy_events: Число policy-решений за прогон.
        events: Строки телеметрии.

    Returns:
        Числовая метрика, доля policy-решений или множество классов реакций.
    """
    if check.observable == _OBSERVABLE_REACTIONS:
        return frozenset(
            e.get("policy_action", "")
            for e in events
            if e.get("policy_action", "")
        )
    if check.observable == _OBSERVABLE_POLICY_RATE:
        return policy_events / check.ticks
    return behavioral_fingerprint(events).metric(check.observable)


def _observable_changed(baseline: object, ablated: object, *, tol: float) -> bool:
    """Изменилось ли наблюдаемое (чистая).

    Args:
        baseline: Наблюдаемое с механизмом.
        ablated: Наблюдаемое без механизма.
        tol: Допуск на изменение числовой наблюдаемой.

    Returns:
        True, если наблюдаемое изменилось.
    """
    if isinstance(baseline, frozenset) and isinstance(ablated, frozenset):
        return baseline != ablated
    return abs(float(baseline) - float(ablated)) > tol  # type: ignore[arg-type]


def _ablation_reason(
    check: AblationCheck,
    baseline: object,
    ablated: object,
    changed: bool,
) -> str:
    """Собрать причину вердикта ablation (чистая)."""
    verdict = "changed" if changed else "unchanged"
    expectation = "change" if check.expect_change else "no change"
    status = "ok" if changed == check.expect_change else "unexpected"
    return (
        f"{check.mechanism.value}: {verdict} (expected {expectation}) — "
        f"{status}; baseline={baseline!r}, ablated={ablated!r}"
    )


def _read_events(log_path: Path) -> list[dict[str, Any]]:
    """Прочитать JSONL телеметрии в список словарей (чистая)."""
    text = log_path.read_text().strip()
    if not text:
        return []
    return [json.loads(line) for line in text.split("\n")]


def _take_view(loop: HostLoop) -> StateView:
    """Снять компактный снимок состояния с loop (чистая)."""
    outcome = loop.last_outcome
    return StateView(
        tick=loop.current_tick,
        f=outcome.result.f if outcome is not None else 0.0,
        valence=outcome.result.valence if outcome is not None else 0.0,
        stress=outcome.result.allostatic_stress if outcome is not None else 0.0,
        gamma=outcome.result.gamma if outcome is not None else 0.0,
        task=loop.active_task(),
        active_columns=loop.pipeline.ensemble.active,
        drift=loop.last_drift,
        partner_trust=loop.last_partner_trust,
        partner_uncertainty=loop.last_partner_uncertainty,
    )


def summarize(results: Sequence[ScenarioResult]) -> Mapping[str, object]:
    """Сводка корпуса: сколько прошло, какие сценарии провалились (чистая).

    Args:
        results: Результаты прогонов.

    Returns:
        Словарь ``{"total", "passed", "failed", "failed_ids"}``.
    """
    failed = [r for r in results if not r.passed]
    return {
        "total": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "failed_ids": tuple(r.scenario.id for r in failed),
    }
