"""Behavioral chain harness — state, decision, intent, actuation, reply.

The behavioral auto-test checks the host's **state channel** link by link
(VALIDATION §7). This module covers all five links plus a separate fidelity
harness for the LLM transducer:

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
* **Link 4 — actuation**: the LLM is called *iff* the decision is to speak, and
  a reply comes back (or the failure is recorded). Checked by *actuation
  invariants* with a ``RecordingLlmClient`` (VALIDATION §7.1, §7.4).
* **Link 5 — reply**: the reply is *structurally* relevant to the intent — its
  class (statement/question/empty) matches the frame's goal. Checked by *reply
  invariants*; semantics is HITL, not a gate (VALIDATION §7.4).

Fidelity harness (VALIDATION §7.6): a *separate* harness for the LLM
transducer. It checks the **order** of a signal, not its absolute value
(masking / inversion / fabrication), with a deterministic lexicon scorer in CI
and opt-in anchor embeddings / a real LLM.

Functional Core (pure, ADR-0004):

* :class:`ReactionClass` — the observable decision class.
* :class:`StateInvariant` — a boundedness/finiteness property of the state.
* :class:`StateView` — a compact snapshot of the state (link 1).
* :class:`IntentInvariant` / :class:`IntentView` / :func:`check_intent` /
  :func:`intent_from_state` — intent grounding (link 3).
* :class:`ActuationInvariant` / :class:`ActuationView` / :func:`check_actuation`
  — actuation invariants (link 4).
* :class:`ReplyInvariant` / :class:`ReplyClass` / :class:`ReplyView` /
  :func:`classify_reply` / :func:`check_reply` — reply structure (link 5).
* :class:`PreconditionKind` / :class:`Precondition` — how much history a
  scenario needs (``born``/``primed(n)``/``matured``; VALIDATION §7.8).
* :class:`Scenario` / :class:`ScenarioResult` — one test cell and its outcome.
* :func:`summarize` / :func:`result_dict` / :func:`report_dict` — a serializable
  run report (CLI ``--behavioral`` / JSON).
* :class:`OperationFact` / :func:`operation_facts` — human-readable decomposition
  of a run into operations (count / reason / expected; §7.1).
* :func:`observed_shares` / :func:`baseline_dict` / :class:`Deviation` /
  :func:`compare_to_baseline` — calibration of observed shares against a baseline
  band (CLI ``--behavioral-baseline`` / ``--behavioral-save-baseline``; §7.1).
* :class:`Ablation` / :class:`AblationCheck` / :class:`AblationResult` —
  attribution checks: disabling a mechanism must *change* the observable
  (VALIDATION §7.5). Non-tautological: if nothing changes, the test was about
  substitution, not the mechanism.
* :class:`ToneAxis` / :class:`DistortionClass` / :class:`ToneScorer` /
  :class:`LexiconToneScorer` / :class:`EmbeddingToneScorer` /
  :class:`FidelityPair` / :class:`FidelityResult` / :func:`check_fidelity` —
  transducer fidelity (VALIDATION §7.6).

Imperative Shell:

* :class:`BehavioralChainRunner` — runs a scenario over ``HostLoop`` with
  deterministic knobs (preset + synthetic clock + fake embedder + fake meter),
  driving policy itself (no LLM, ADR-0007), mirroring ``ChatSession``'s decision
  path. Link 4 routes speaking decisions through ``SpeechController.respond``
  with a ``RecordingLlmClient`` (deterministic fake LLM), so the call/no-call is
  observable. Same seed → same result (VALIDATION §7.7). Preconditions (VALIDATION
  §7.8): ``run_all(precondition=...)`` keeps applicable scenarios and
  ``run_matured`` is the long-horizon entry; ``primed(n)`` warms up the first n
  messages before measurement.
* :class:`RecordingLlmClient` — a fake LLM that records every call and returns a
  goal-structured deterministic reply.
* :class:`FidelityHarness` — drives an injected responder over fidelity pairs.

Not covered yet (by design): ToM (``partner`` is ``None`` → S4-compat).
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from src.config import HostConfig, load_preset
from src.core.policy import Action, PolicyTrace, select_action
from src.host.fingerprint import FINGERPRINT_METRICS, behavioral_fingerprint
from src.host.loop import HostLoop, build_host_loop
from src.host.sensitivity import DeterministicMeter
from src.memory.embedder import Embedder
from src.speech.controller import SpeechController, SpeechDecision
from src.speech.intent import (
    GOAL_INSTRUCTIONS,
    IntentFrame,
    build_intent_frame,
    describe_affect,
    escape_hatch_message,
    goal_for_action,
)
from src.speech.llm import LlmError

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


class ActuationInvariant(Enum):
    """Наблюдаемое свойство актюации (звено 4, VALIDATION §7.1).

    Актюация — это вызов LLM. Проверяется не текст ответа, а *факт* вызова:
        LLM_CALLED_IFF_SPEAK — LLM вызван тогда и только тогда, когда решение
            «говорить» (молчание не должно жечь дорогой вызов);
        RESPONSE_RETURNED — при вызове вернулся непустой ответ либо сбой
            зафиксирован (хост не падает, VALIDATION §7.3);
        FRAME_GROUNDED — цель, переданная в реплику, следует из решения policy
            (маппинг ``Action → goal`` соблюдён до самого LLM).
    """

    LLM_CALLED_IFF_SPEAK = "llm_called_iff_speak"
    RESPONSE_RETURNED = "response_returned"
    FRAME_GROUNDED = "frame_grounded"


_DEFAULT_ACTUATION_INVARIANTS: tuple[ActuationInvariant, ...] = (
    ActuationInvariant.LLM_CALLED_IFF_SPEAK,
    ActuationInvariant.RESPONSE_RETURNED,
    ActuationInvariant.FRAME_GROUNDED,
)


@dataclass(frozen=True)
class ActuationView:
    """Снимок звена 4: решение, факт вызова LLM и ответ.

    Attributes:
        decision: Решение о речи, переданное в контроллер (None → не решали).
        called: Был ли фактически вызван LLM.
        response: Текст ответа (None — не отвечали или сбой).
        error: Зафиксирован ли сбой LLM (``LlmError``).
        escape_hatch: Реплика отдана дешёвым шаблоном (без LLM, S4-долг).
        frame_goal: Цель, с которой строился IntentFrame для реплики.
        action: Решение policy (None → S3-фолбэк).
    """

    decision: SpeechDecision | None
    called: bool
    response: str | None
    error: bool
    frame_goal: str | None
    action: Action | None
    escape_hatch: bool = False


def check_actuation(
    view: ActuationView, invariants: Sequence[ActuationInvariant]
) -> tuple[str, ...]:
    """Проверить инварианты актюации; вернуть имена нарушенных (чистая).

    Args:
        view: Снимок актюации.
        invariants: Проверяемые инварианты.

    Returns:
        Кортеж имён нарушенных инвариантов (пусто → всё выполнено).
    """
    speak = view.decision is not None and view.decision.speak
    # Escape hatch — отдельный дешёвый путь: право голоса без LLM-вызова.
    expect_call = speak and not view.escape_hatch
    violated: list[str] = []
    for invariant in invariants:
        if invariant is ActuationInvariant.LLM_CALLED_IFF_SPEAK:
            ok = view.called == expect_call
        elif invariant is ActuationInvariant.RESPONSE_RETURNED:
            ok = (
                not view.called
                or view.error
                or bool(view.response and view.response.strip())
            )
        elif invariant is ActuationInvariant.FRAME_GROUNDED:
            ok = (
                view.escape_hatch
                or view.action is None
                or view.frame_goal == goal_for_action(view.action)
            )
        else:  # pragma: no cover — все члены перечислены
            ok = True
        if not ok:
            violated.append(invariant.value)
    return tuple(violated)


class ReplyClass(Enum):
    """Структурный класс реплики (звено 5, VALIDATION §7.4).

    Классификация — по поверхностной форме, а не по тексту: пусто/вопрос/
    утверждение. Это позволяет машинно сверять реплику с целью интента без
    семантики (семантику судит HITL, ADR-0007).
    """

    EMPTY = "empty"
    QUESTION = "question"
    STATEMENT = "statement"


# Какие структурные классы допустимы для цели. Вопрос требуется там, где цель
# — уточнение (identify_partner/explore); молчание — только пустая реплика.
_GOAL_REPLY_CLASSES: dict[str, frozenset[ReplyClass]] = {
    "respond": frozenset({ReplyClass.STATEMENT, ReplyClass.QUESTION}),
    "initiative": frozenset({ReplyClass.STATEMENT, ReplyClass.QUESTION}),
    "identify_partner": frozenset({ReplyClass.QUESTION}),
    "explore": frozenset({ReplyClass.QUESTION}),
    "silent": frozenset({ReplyClass.EMPTY}),
}


class ReplyInvariant(Enum):
    """Наблюдаемое свойство реплики (звено 5, VALIDATION §7.4).

        CLASS_MATCHES_GOAL — структурный класс реплики допустим для цели интента;
        NONEMPTY — при решении «говорить» реплика непуста.
    """

    CLASS_MATCHES_GOAL = "class_matches_goal"
    NONEMPTY = "nonempty"


_DEFAULT_REPLY_INVARIANTS: tuple[ReplyInvariant, ...] = (
    ReplyInvariant.CLASS_MATCHES_GOAL,
    ReplyInvariant.NONEMPTY,
)


@dataclass(frozen=True)
class ReplyView:
    """Снимок звена 5: интент + фактическая реплика.

    Attributes:
        frame: IntentFrame, с которым строилась реплика.
        text: Текст реплики (None — не отвечали).
    """

    frame: IntentFrame
    text: str | None


def classify_reply(text: str | None) -> ReplyClass:
    """Классифицировать реплику по поверхностной форме (чистая).

    Args:
        text: Текст реплики (None/пусто → ``EMPTY``).

    Returns:
        ``EMPTY`` для пустой, ``QUESTION`` при знаке вопроса, иначе
        ``STATEMENT``.
    """
    if text is None or not text.strip():
        return ReplyClass.EMPTY
    if "?" in text:
        return ReplyClass.QUESTION
    return ReplyClass.STATEMENT


def check_reply(
    view: ReplyView, invariants: Sequence[ReplyInvariant]
) -> tuple[str, ...]:
    """Проверить структурную релевантность реплики интенту (чистая).

    Args:
        view: Снимок реплики.
        invariants: Проверяемые инварианты.

    Returns:
        Кортеж имён нарушенных инвариантов (пусто → всё выполнено).
    """
    reply_class = classify_reply(view.text)
    allowed = _GOAL_REPLY_CLASSES.get(view.frame.goal)
    violated: list[str] = []
    for invariant in invariants:
        if invariant is ReplyInvariant.CLASS_MATCHES_GOAL:
            ok = allowed is None or reply_class in allowed
        elif invariant is ReplyInvariant.NONEMPTY:
            ok = reply_class is not ReplyClass.EMPTY
        else:  # pragma: no cover — все члены перечислены
            ok = True
        if not ok:
            violated.append(invariant.value)
    return tuple(violated)


@dataclass
class RecordingLlmClient:
    """Fake-LLM для звеньев 4–5: пишет вызовы и строит реплику по цели.

    Детерминированный (без сети). Возвращает поверхностную форму, согласованную
    с целью из system-промпта: для целей-уточнений (``identify_partner``,
    ``explore``) — вопрос, иначе — утверждение. Записывает ``messages`` каждого
    вызова, чтобы звено 4 видело факт вызова, а звено 5 — реплику.

    Attributes:
        calls: Сохранённые ``messages`` по каждому вызову ``reply``.
    """

    calls: list[list[dict[str, Any]]] = field(
        default_factory=list[list[dict[str, Any]]]
    )

    def reply(self, messages: list[dict[str, Any]], max_tokens: int = 256) -> str:
        """Записать вызов и вернуть реплику, согласованную с целью.

        Args:
            messages: Chat messages.
            max_tokens: Лимит токенов (не влияет на форму в fake-режиме).

        Returns:
            Непустой детерминированный текст.
        """
        self.calls.append(messages)
        system = messages[0]["content"] if messages else ""
        if goal_instruction_present(system, "identify_partner"):
            return "Как я могу к тебе обращаться?"
        if goal_instruction_present(system, "explore"):
            return "Уточни, пожалуйста, что именно ты имеешь в виду?"
        return "Понял, отвечаю."

    @property
    def call_count(self) -> int:
        """Число вызовов LLM за прогон."""
        return len(self.calls)


def goal_instruction_present(system: str, goal: str) -> bool:
    """Есть ли в system-промпте инструкция заданной цели (чистая).

    Args:
        system: System-сообщение.
        goal: Цель (``respond``/``initiative``/``identify_partner``/...).

    Returns:
        True, если инструкция цели присутствует в промпте.
    """
    instruction = GOAL_INSTRUCTIONS.get(goal)
    return bool(instruction) and instruction in system


@dataclass
class FailingLlmClient:
    """Fake-LLM, всегда бросающий ``LlmError`` (проверка graceful degradation).

    Attributes:
        calls: Сохранённые ``messages`` (вызов записывается до падения).
    """

    calls: list[list[dict[str, Any]]] = field(
        default_factory=list[list[dict[str, Any]]]
    )

    def reply(self, messages: list[dict[str, Any]], max_tokens: int = 256) -> str:
        """Записать вызов и упасть (сбой LLM).

        Args:
            messages: Chat messages.
            max_tokens: Лимит токенов (не используется).

        Raises:
            LlmError: Всегда.
        """
        self.calls.append(messages)
        raise LlmError("deterministic failure")


class PreconditionKind(Enum):
    """Объём истории, при котором сценарий имеет смысл (VALIDATION §7.8).

        BORN — с нуля (ворота, детерминизм);
        PRIMED — прогрев N реплик в том же прогоне до замера;
        MATURED — длинный прогон (отдельный harness, как C10).
    """

    BORN = "born"
    PRIMED = "primed"
    MATURED = "matured"


@dataclass(frozen=True)
class Precondition:
    """Предусловие сценария: сколько истории нужно до замера (VALIDATION §7.8).

    Сценарий декларирует объём истории, при котором он осмыслен; runner
    выбирает применимые (:meth:`applies_to`). ``warmup`` задан только для
    ``PRIMED`` — число прогревочных реплик.

    Attributes:
        kind: Вид предусловия.
        warmup: Число прогревочных реплик (``PRIMED``; иначе 0).

    Raises:
        ValueError: Если ``warmup`` отрицателен, задан вне ``PRIMED`` или
            ``PRIMED`` без прогрева.
    """

    kind: PreconditionKind = PreconditionKind.BORN
    warmup: int = 0

    def __post_init__(self) -> None:
        if self.warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {self.warmup}")
        if self.kind is PreconditionKind.PRIMED:
            if self.warmup < 1:
                raise ValueError("primed precondition requires warmup >= 1")
        elif self.warmup != 0:
            raise ValueError(
                f"warmup is only valid for primed, got {self.kind.value}"
            )

    @classmethod
    def born(cls) -> Precondition:
        """Предусловие «с нуля» (ворота, детерминизм)."""
        return cls(PreconditionKind.BORN)

    @classmethod
    def primed(cls, warmup: int) -> Precondition:
        """Предусловие с прогревом ``warmup`` реплик.

        Args:
            warmup: Число прогревочных реплик (>= 1).

        Returns:
            Предусловие ``PRIMED``.
        """
        return cls(PreconditionKind.PRIMED, warmup)

    @classmethod
    def matured(cls) -> Precondition:
        """Предусловие длинного прогона (отдельный harness)."""
        return cls(PreconditionKind.MATURED)

    def applies_to(self, run: Precondition) -> bool:
        """Применим ли сценарий при прогоне с предусловием ``run``.

        Сценарий применим, если требуемый объём истории не превышает
        предоставленный прогоном: ``born`` — только ``born``; ``primed(n)`` —
        ``primed(m)`` при ``m >= n``; ``matured`` — только ``matured``.

        Args:
            run: Предусловие прогона.

        Returns:
            True, если сценарий можно запускать в этом прогоне.
        """
        if self.kind is not run.kind:
            return False
        if self.kind is PreconditionKind.PRIMED:
            return run.warmup >= self.warmup
        return True

    def __str__(self) -> str:
        """Человекочитаемая форма (``born`` | ``primed(n)`` | ``matured``)."""
        if self.kind is PreconditionKind.PRIMED:
            return f"primed({self.warmup})"
        return self.kind.value


@dataclass(frozen=True)
class Scenario:
    """Одна ячейка поведенческого теста (сценарий → ожидаемый класс).

    Attributes:
        id: Идентификатор (например ``state.bounds.baseline``).
        link: Проверяемое звено (``state`` | ``decision`` | ``intent`` |
            ``actuation`` | ``reply``).
        preset: Имя пресета (ручки заморожены).
        seed: Зерно детерминированных провайдеров.
        ticks: Число тиков прогона.
        messages: Скрипт сообщений ``(tick, text)``.
        expect_reaction: Ожидаемый класс реакции (None → не проверяется).
        state_invariants: Проверяемые инварианты состояния.
        precondition: Предусловие объёма истории (VALIDATION §7.8).

    Raises:
        ValueError: Если id/звено пусты, ticks < 1 или у ``primed(n)`` нет
            измеряемого сообщения после прогрева.
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
    actuation_invariants: tuple[ActuationInvariant, ...] = (
        _DEFAULT_ACTUATION_INVARIANTS
    )
    reply_invariants: tuple[ReplyInvariant, ...] = _DEFAULT_REPLY_INVARIANTS
    precondition: Precondition = Precondition()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("scenario id must not be empty")
        if self.link not in (
            "state",
            "decision",
            "intent",
            "actuation",
            "reply",
        ):
            raise ValueError(
                f"link must be 'state'|'decision'|'intent'|'actuation'|'reply', "
                f"got {self.link!r}"
            )
        if self.ticks < 1:
            raise ValueError(f"ticks must be >= 1, got {self.ticks}")
        if self.precondition.kind is PreconditionKind.PRIMED and (
            len(self.messages) <= self.precondition.warmup
        ):
            raise ValueError(
                f"primed({self.precondition.warmup}) scenario needs a measured "
                f"message after warm-up, got {len(self.messages)} message(s)"
            )

    def measure_from(self) -> int:
        """Тик, с которого начинается замер (после прогрева; VALIDATION §7.8).

        Для ``born``/``matured`` — 0; для ``primed(n)`` — тик (n+1)-го
        сообщения (первые n сообщений — прогрев).

        Returns:
            Номер тика начала замера.
        """
        if self.precondition.kind is not PreconditionKind.PRIMED:
            return 0
        ticks = sorted(tick for tick, _ in self.messages)
        return ticks[self.precondition.warmup]


@dataclass(frozen=True)
class ScenarioResult:
    """Итог прогона сценария.

    Attributes:
        scenario: Прогнанный сценарий.
        reactions: Наблюдаемые классы реакций по тикам (в порядке тиков).
        violations: Имена нарушенных инвариантов состояния (уникальные).
        intent_violations: Имена нарушенных инвариантов интента (уникальные).
        actuation_violations: Имена нарушенных инвариантов актюации (уникальные).
        reply_violations: Имена нарушенных инвариантов реплики (уникальные).
        llm_calls: Число вызовов LLM за прогон (звено 4).
        passed: Выполнен ли сценарий (нет нарушений + ожидание совпало).
        reason: Причина вердикта (для диагностики).
        throttled_calls: Число говорящих решений, отсечённых throttle
            (llm_gate) — вычитаемое в декомпозиции ``llm_calls`` (§7.1).
    """

    scenario: Scenario
    reactions: tuple[ReactionClass, ...]
    violations: tuple[str, ...]
    intent_violations: tuple[str, ...]
    actuation_violations: tuple[str, ...]
    reply_violations: tuple[str, ...]
    llm_calls: int
    passed: bool
    reason: str
    throttled_calls: int = 0


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
    """Встроенный корпус сценариев звеньев 1–5 + предусловия (VALIDATION §7).

    Returns:
        Кортеж :class:`Scenario`. Сценарии, бессмысленные без истории,
        объявляют ``primed``/``matured`` и на ``born`` не запускаются
        (VALIDATION §7.8).
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
        # Звено 4: новое сообщение → решение «говорить» → LLM вызван ровно
        # один раз (актюация), ответ вернулся.
        Scenario(
            id="actuation.respond.calls_llm",
            link="actuation",
            preset="dialogue",
            ticks=120,
            messages=((0, "привет"),),
        ),
        # Звено 4: без сообщения возможны инициатива/исследование — инвариант
        # «LLM вызван ⇔ решение говорить» проверяется на каждом тике.
        Scenario(
            id="actuation.baseline.invariants",
            link="actuation",
            preset="baseline",
            ticks=120,
        ),
        # Звено 5: реплика структурно релевантна цели интента.
        Scenario(
            id="reply.respond.class_matches",
            link="reply",
            preset="dialogue",
            ticks=120,
            messages=((0, "привет"),),
        ),
        # Предусловие primed: первая реплика — прогрев, замер со второй.
        # Это проверка формата прогона (исправность канала при ненулевой
        # истории), а не объекта накопления: узнавание/recall — mature-harness
        # (VALIDATION §7.8).
        Scenario(
            id="precondition.primed.state",
            link="state",
            preset="dialogue",
            ticks=120,
            messages=((0, "привет"), (60, "это снова я")),
            precondition=Precondition.primed(1),
        ),
        # Предусловие matured: длинный прогон (long-horizon), отдельный harness.
        Scenario(
            id="precondition.matured.long_horizon",
            link="state",
            preset="long-horizon",
            ticks=1500,
            precondition=Precondition.matured(),
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
    """Shell: прогон сценариев поверх ``HostLoop``, сбор наблюдаемых по звеньям.

    Attributes:
        workdir: Каталог для логов/БД (детерминизм не зависит от пути).
        llm_factory: Фабрика LLM-клиента на прогон (звенья 4–5). По умолчанию
            ``RecordingLlmClient`` (детерминированный fake, пишет вызовы);
            тесты подменяют на сбойный клиент.
    """

    def __init__(
        self,
        *,
        workdir: Path | None = None,
        llm_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.workdir = (
            Path(workdir)
            if workdir is not None
            else Path(tempfile.mkdtemp(prefix="behavioral-chain-"))
        )
        self.llm_factory = llm_factory if llm_factory is not None else RecordingLlmClient
        self._counter = 0

    def run(self, scenario: Scenario) -> ScenarioResult:
        """Прогнать один сценарий и вернуть результат.

        Args:
            scenario: Сценарий.

        Returns:
            ScenarioResult с наблюдаемыми по звеньям и вердиктом.
        """
        config = self._make_config(scenario)
        obs = self._run_config(config, scenario)
        unique_violations = tuple(dict.fromkeys(obs.violations))
        unique_intent = tuple(dict.fromkeys(obs.intent_violations))
        unique_actuation = tuple(dict.fromkeys(obs.actuation_violations))
        unique_reply = tuple(dict.fromkeys(obs.reply_violations))
        expected_ok = (
            scenario.expect_reaction is None
            or scenario.expect_reaction in obs.reactions
        )
        passed = (
            not unique_violations
            and not unique_intent
            and not unique_actuation
            and not unique_reply
            and expected_ok
        )
        reason = self._reason(
            scenario,
            obs.reactions,
            unique_violations,
            unique_intent,
            unique_actuation,
            unique_reply,
            expected_ok,
        )
        return ScenarioResult(
            scenario=scenario,
            reactions=tuple(obs.reactions),
            violations=unique_violations,
            intent_violations=unique_intent,
            actuation_violations=unique_actuation,
            reply_violations=unique_reply,
            llm_calls=obs.llm_calls,
            passed=passed,
            reason=reason,
            throttled_calls=obs.throttled_calls,
        )

    def run_all(
        self,
        scenarios: Sequence[Scenario] | None = None,
        *,
        precondition: Precondition | None = None,
    ) -> list[ScenarioResult]:
        """Прогнать корпус сценариев (по умолчанию — встроенный).

        Args:
            scenarios: Сценарии (None → :func:`default_scenarios`).
            precondition: Если задано, оставить только сценарии, применимые к
                этому предусловию (:meth:`Precondition.applies_to`; §7.8).

        Returns:
            Список :class:`ScenarioResult` в порядке сценариев.
        """
        corpus = default_scenarios() if scenarios is None else scenarios
        if precondition is not None:
            corpus = [
                s for s in corpus if s.precondition.applies_to(precondition)
            ]
        return [self.run(scenario) for scenario in corpus]

    def run_matured(
        self, scenarios: Sequence[Scenario] | None = None
    ) -> list[ScenarioResult]:
        """Отдельный длинный прогон: только ``matured``-сценарии (§7.8).

        Args:
            scenarios: Сценарии (None → :func:`default_scenarios`).

        Returns:
            Список :class:`ScenarioResult` в порядке сценариев.
        """
        return self.run_all(scenarios, precondition=Precondition.matured())

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

        base_obs = self._run_config(base_config, scenario)
        abl_obs = self._run_config(ablated_config, scenario)

        baseline = _observable(check, base_obs.policy_events, base_obs.events)
        ablated = _observable(check, abl_obs.policy_events, abl_obs.events)
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

    def _run_config(self, config: HostConfig, scenario: Scenario) -> _Observables:
        """Прогнать loop под готовым конфигом и собрать наблюдаемые.

        Returns:
            :class:`_Observables` по всем звеньям.
        """
        loop = build_host_loop(
            config, meter=DeterministicMeter(), messages=scenario.messages
        )
        llm = self.llm_factory()
        # Recall прецедентов намеренно не подключён: звенья 4–5 проверяют
        # актюацию/реплику, а содержание памяти покрыто ablation (§7.5).
        # MemoryRouter.store не типизируется pyright (пересечение Protocol'ов
        # в memory/router.py — предсуществующий дефект вне этого модуля).
        controller = SpeechController(
            llm=llm,
            f_threshold=config.speech.f_threshold,
            default_register=config.speech.default_register,
        )
        try:
            obs = self._drive(loop, config, scenario, controller, llm)
            obs.events = _read_events(Path(config.log_path))
        finally:
            loop.close()
        return obs

    @staticmethod
    def _drive(
        loop: HostLoop,
        config: HostConfig,
        scenario: Scenario,
        controller: SpeechController,
        llm: Any,
    ) -> _Observables:
        """Прогнать тики, ведя policy вручную (как ``ChatSession``).

        Новое сообщение на тике ``t`` → решение с ``has_new_message=True``;
        иначе — с ``False`` (возможны инициатива/исследование). Решения звена 2
        принимаются policy; при решении «говорить» звено 4 маршрутизирует реплику
        через ``SpeechController.respond`` (LLM — fake, ADR-0007), звено 5
        проверяет структурную форму ответа. При ``policy.enabled=False`` решения
        не принимаются (ablation). На каждом тике интент собирается из состояния
        и решения (звено 3) и проверяется на grounding.

        Returns:
            :class:`_Observables` по всем звеньям.
        """
        message_ticks = {tick for tick, _ in scenario.messages}
        measure_from = scenario.measure_from()
        obs = _Observables()
        for tick in range(scenario.ticks):
            loop.step_once(tick)
            # Прогрев (primed): loop и policy прогоняются, история копится, но
            # наблюдаемые не записываются до начала замера (VALIDATION §7.8).
            measuring = tick >= measure_from
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
                if measuring:
                    obs.reactions.append(
                        classify_reaction(
                            trace, escape_hatch=loop.escape_hatch_active
                        )
                    )
                    obs.policy_events += 1
                BehavioralChainRunner._drive_actuation(
                    loop, scenario, controller, llm, trace, action, obs, measuring
                )
            if measuring:
                view = _take_view(loop)
                obs.violations.extend(
                    check_state(view, scenario.state_invariants)
                )
                intent_view = intent_from_state(view, action)
                obs.intent_violations.extend(
                    check_intent(intent_view, scenario.intent_invariants)
                )
        return obs

    @staticmethod
    def _drive_actuation(
        loop: HostLoop,
        scenario: Scenario,
        controller: SpeechController,
        llm: Any,
        trace: PolicyTrace,
        action: Action | None,
        obs: _Observables,
        observe: bool = True,
    ) -> None:
        """Звенья 4–5: маршрутизировать реплику и проверить её форму.

        Решение «говорить» повторяет ``ChatSession``: не ``SILENT`` и не
        удержанный ``llm_gate`` (throttle). Escape hatch (S4-долг) отдаётся
        дешёвым шаблоном без LLM; иначе при решении «говорить» вызывается
        контроллер. Реплика проверяется на структурную релевантность интенту
        (звено 5) только когда она фактически получена — сбой LLM относится к
        звену 4, а не к звено 5.

        Args:
            loop: Host loop.
            scenario: Сценарий.
            controller: Контроллер речи.
            llm: LLM-клиент прогона (для чтения факта вызова).
            trace: Трасса решения policy.
            action: Решение policy (None → S3-фолбэк).
            obs: Аккумулятор наблюдаемых.
            observe: Записывать ли наблюдаемые (False — прогрев; §7.8).
        """
        view = _take_view(loop)
        goal = goal_for_action(action) if action is not None else "respond"
        has_new_message = loop.current_tick in {t for t, _ in scenario.messages}
        throttled = loop.last_throttle.llm_gate and not has_new_message
        speak = trace.chosen is not Action.SILENT and not throttled
        # Декомпозиция llm_calls: говорящее решение, отсечённое throttle
        # (llm_gate) — вычитаемое к числу вызовов (§7.1).
        if (
            observe
            and throttled
            and not loop.escape_hatch_active
            and trace.chosen is not Action.SILENT
        ):
            obs.throttled_calls += 1
        decision = SpeechDecision(speak=speak, reason=trace.chosen.value)
        before = _llm_calls(llm)
        response: str | None = None
        escape = False

        if loop.escape_hatch_active:
            escape = True
            response = escape_hatch_message(task=view.task, stress=view.stress)
            loop.mark_spoke()
        elif speak:
            response = controller.respond(
                user_text=_current_text(loop, scenario),
                f=view.f,
                valence=view.valence,
                stress=view.stress,
                task=view.task,
                new_message=has_new_message,
                goal=goal,
                decision=decision,
            )
            if response is not None:
                loop.mark_spoke()

        called = _llm_calls(llm) > before
        error = called and response is None
        frame_goal = None if escape else goal
        actuation_view = ActuationView(
            decision=decision,
            called=called,
            response=response,
            error=error,
            frame_goal=frame_goal,
            action=action,
            escape_hatch=escape,
        )
        if observe:
            obs.actuation_violations.extend(
                check_actuation(actuation_view, scenario.actuation_invariants)
            )
            obs.llm_calls += 1 if called else 0

        if response is not None and observe:
            frame = build_intent_frame(
                f=view.f,
                valence=view.valence,
                stress=view.stress,
                task=view.task,
                goal=goal,
            )
            obs.reply_violations.extend(
                check_reply(
                    ReplyView(frame=frame, text=response), scenario.reply_invariants
                )
            )

    @staticmethod
    def _reason(
        scenario: Scenario,
        reactions: Sequence[ReactionClass],
        violations: tuple[str, ...],
        intent_violations: tuple[str, ...],
        actuation_violations: tuple[str, ...],
        reply_violations: tuple[str, ...],
        expected_ok: bool,
    ) -> str:
        """Собрать причину вердикта (для журнала/диагностики)."""
        if violations:
            return f"state invariant violated: {', '.join(violations)}"
        if intent_violations:
            return f"intent invariant violated: {', '.join(intent_violations)}"
        if actuation_violations:
            return f"actuation invariant violated: {', '.join(actuation_violations)}"
        if reply_violations:
            return f"reply invariant violated: {', '.join(reply_violations)}"
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


def _llm_calls(llm: Any) -> int:
    """Число вызовов LLM-клиента (0, если клиент не ведёт счётчик).

    Args:
        llm: LLM-клиент (``RecordingLlmClient``/``FailingLlmClient``).

    Returns:
        Число вызовов ``reply``.
    """
    calls = getattr(llm, "calls", None)
    return len(calls) if calls is not None else 0


def _current_text(loop: HostLoop, scenario: Scenario) -> str:
    """Текущее сообщение оператора на последнем тике (для recall).

    Args:
        loop: Host loop.
        scenario: Сценарий (скрипт сообщений).

    Returns:
        Текст сообщения на текущем тике или "" (нет сообщения).
    """
    tick = loop.current_tick
    for message_tick, text in scenario.messages:
        if message_tick == tick:
            return text
    return ""


@dataclass
class _Observables:
    """Аккумулятор наблюдаемых по звеньям за один прогон (Shell-internal).

    Attributes:
        reactions: Классы реакций (звено 2).
        violations: Нарушения инвариантов состояния (звено 1).
        intent_violations: Нарушения grounding интента (звено 3).
        actuation_violations: Нарушения инвариантов актюации (звено 4).
        reply_violations: Нарушения инвариантов реплики (звено 5).
        policy_events: Число policy-решений за прогон.
        llm_calls: Число вызовов LLM за прогон.
        throttled_calls: Число говорящих решений, отсечённых throttle.
        events: Строки телеметрии.
    """

    reactions: list[ReactionClass] = field(
        default_factory=list[ReactionClass]
    )
    violations: list[str] = field(default_factory=list[str])
    intent_violations: list[str] = field(default_factory=list[str])
    actuation_violations: list[str] = field(default_factory=list[str])
    reply_violations: list[str] = field(default_factory=list[str])
    policy_events: int = 0
    llm_calls: int = 0
    throttled_calls: int = 0
    events: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])


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


def result_dict(result: ScenarioResult) -> dict[str, Any]:
    """Сериализуемый снимок результата сценария (чистая).

    Args:
        result: Результат прогона.

    Returns:
        Словарь с id/звеном/предусловием, вердиктом, наблюдаемыми и причинами.
    """
    return {
        "id": result.scenario.id,
        "link": result.scenario.link,
        "preset": result.scenario.preset,
        "precondition": str(result.scenario.precondition),
        "passed": result.passed,
        "reason": result.reason,
        "reactions": [reaction.value for reaction in result.reactions],
        "llm_calls": result.llm_calls,
        "throttled_calls": result.throttled_calls,
        "state_violations": list(result.violations),
        "intent_violations": list(result.intent_violations),
        "actuation_violations": list(result.actuation_violations),
        "reply_violations": list(result.reply_violations),
    }


def report_dict(results: Sequence[ScenarioResult]) -> dict[str, Any]:
    """Сериализуемый отчёт прогона: сводка + по сценариям (чистая).

    Форма отчёта для CLI (`--behavioral-json`) и будущего визуализатора;
    JSON-совместим (без tuple/Enum).

    Args:
        results: Результаты прогонов.

    Returns:
        ``{"summary": {...}, "scenarios": [...]}``.
    """
    summary = summarize(results)
    return {
        "summary": {
            "total": summary["total"],
            "passed": summary["passed"],
            "failed": summary["failed"],
            "failed_ids": list(summary["failed_ids"]),  # type: ignore[arg-type]
        },
        "scenarios": [result_dict(result) for result in results],
    }


# Реакции-«говорение»: решение policy породить реплику (LLM вызывается, если
# не отсечено throttle и не подменено escape hatch). Основа декомпозиции
# ``llm_calls`` (VALIDATION §7.1).
_SPEAKING_REACTIONS: frozenset[ReactionClass] = frozenset(
    {
        ReactionClass.RESPOND,
        ReactionClass.INITIATIVE,
        ReactionClass.IDENTIFY_PARTNER,
        ReactionClass.EXPLORE,
    }
)

# Порядок операций в отчёте: что хост делал (речь) → тишина → escape.
_REACTION_ORDER: tuple[ReactionClass, ...] = (
    ReactionClass.RESPOND,
    ReactionClass.INITIATIVE,
    ReactionClass.IDENTIFY_PARTNER,
    ReactionClass.EXPLORE,
    ReactionClass.SILENT,
    ReactionClass.ESCAPE_HATCH,
)


@dataclass(frozen=True)
class OperationFact:
    """Одна наблюдаемая «операция» хоста за прогон (декомпозиция, §7.1).

    Человекочитаемая единица отчёта: что хост сделал, сколько раз, почему и
    сколько *ожидалось* (для инвариантов). Чистая структура — рендер в текст
    живёт в CLI.

    Attributes:
        operation: Машинный ключ операции (``silent``, ``llm_call``, ...).
        count: Сколько раз операция выполнена за замер.
        reason: Причина ("" → без причины).
        expected: Ожидаемое число для инварианта (None → наблюдаемое без
            точного эталона; калибруется, а не проверяется).
    """

    operation: str
    count: int
    reason: str = ""
    expected: int | None = None

    @property
    def holds(self) -> bool:
        """Совпадает ли факт с ожидаемым (True, если эталона нет)."""
        return self.expected is None or self.count == self.expected


def operation_facts(result: ScenarioResult) -> tuple[OperationFact, ...]:
    """Разложить прогон на наблюдаемые операции с ожидаемым (чистая, §7.1).

    Декомпозиция делает числа интерпретируемыми: ``llm_calls`` выводится из
    числа говорящих решений за вычетом throttle (llm_gate). Escape hatch —
    отдельная операция (LLM не вызывается), поэтому в говорящие не входит.

    Args:
        result: Результат прогона.

    Returns:
        Кортеж :class:`OperationFact` в стабильном порядке: решения policy
        (классы реакций), затем ``llm_call`` и ``throttled``.
    """
    counts: dict[ReactionClass, int] = {}
    for reaction in result.reactions:
        counts[reaction] = counts.get(reaction, 0) + 1

    facts: list[OperationFact] = []
    for reaction in _REACTION_ORDER:  # речь → тишина → escape (для чтения)
        count = counts.get(reaction, 0)
        if count:
            facts.append(OperationFact(operation=reaction.value, count=count))

    speaking = sum(counts.get(r, 0) for r in _SPEAKING_REACTIONS)
    expected_llm = max(0, speaking - result.throttled_calls)
    decomposition = f"= {speaking} говорящих решений"
    if result.throttled_calls:
        decomposition += f" − {result.throttled_calls} throttle"
    facts.append(
        OperationFact(
            operation="llm_call",
            count=result.llm_calls,
            reason=decomposition,
            expected=expected_llm,
        )
    )
    if result.throttled_calls:
        facts.append(
            OperationFact(
                operation="throttled",
                count=result.throttled_calls,
                reason="llm_gate: окно тишины после инициативы",
            )
        )
    return tuple(facts)


def observed_shares(result: ScenarioResult) -> dict[str, float]:
    """Доли наблюдаемых операций без точного эталона (чистая, §7.1).

    Наблюдаемые (доля речи/escape/throttle) не имеют выводимого «должно быть»:
    их число калибруется по эталону (:func:`compare_to_baseline`), а не
    проверяется. Доля считается от измеренных тиков (без прогрева; §7.8).

    Args:
        result: Результат прогона.

    Returns:
        Отображение ``операция → доля`` для фактов без ``expected``.
    """
    measured = result.scenario.ticks - result.scenario.measure_from()
    return {
        fact.operation: (fact.count / measured if measured else 0.0)
        for fact in operation_facts(result)
        if fact.expected is None
    }


def baseline_dict(results: Sequence[ScenarioResult]) -> dict[str, Any]:
    """Сериализуемый эталон наблюдаемых долей (чистая, §7.1).

    Эталон — снимок наблюдаемых долей по сценариям, снятый с принятого прогона.
    Хранится как JSON (``--behavioral-save-baseline``) и служит полосой для
    сравнения (``--behavioral-baseline``). Сравнивать следует внутри одного
    предусловия (VALIDATION §7.7).

    Args:
        results: Результаты прогонов.

    Returns:
        ``{"version", "scenarios": {id: {operation: share}}}`` (JSON-совместимо).
    """
    return {
        "version": 1,
        "scenarios": {
            result.scenario.id: observed_shares(result) for result in results
        },
    }


@dataclass(frozen=True)
class Deviation:
    """Отклонение наблюдаемой доли от эталона (чистая, §7.1).

    Наблюдаемое калибруется по **полосе**, а не проверяется абсолютом: выход за
    полосу — сигнал пересмотреть эталон (развитие) или поймать overfit
    (VALIDATION §7.7), но не автоматический провал ворот.

    Attributes:
        scenario_id: Идентификатор сценария.
        operation: Машинный ключ операции.
        observed: Наблюдаемая доля в текущем прогоне.
        baseline: Эталонная доля.
        band: Полуширина допустимой полосы.
    """

    scenario_id: str
    operation: str
    observed: float
    baseline: float
    band: float

    @property
    def delta(self) -> float:
        """Смещение относительно эталона (``observed - baseline``)."""
        return self.observed - self.baseline

    @property
    def within_band(self) -> bool:
        """Укладывается ли смещение в полосу ``±band``."""
        return abs(self.delta) <= self.band


def compare_to_baseline(
    results: Sequence[ScenarioResult],
    baseline: Mapping[str, Any],
    *,
    band: float = 0.05,
) -> tuple[Deviation, ...]:
    """Сравнить наблюдаемые доли с эталоном по полосе (чистая, §7.1).

    Сравнивается объединение операций прогона и эталона: отсутствующая операция
    считается долей 0.0. Так ловится и **появление**, и **исчезновение**
    поведения, а не только сдвиг уже наблюдаемого.

    Args:
        results: Результаты текущего прогона.
        baseline: Разобранный эталон (:func:`baseline_dict`).
        band: Полуширина допустимой полосы (доля; по умолчанию 0.05).

    Returns:
        Кортеж :class:`Deviation` для сценариев, присутствующих в эталоне
        (порядок — по сценариям, затем по операциям).
    """
    scenarios = baseline.get("scenarios", {})
    deviations: list[Deviation] = []
    for result in results:
        expected = scenarios.get(result.scenario.id)
        if not expected:
            continue
        observed = observed_shares(result)
        for operation in dict.fromkeys([*observed, *expected]):
            deviations.append(
                Deviation(
                    scenario_id=result.scenario.id,
                    operation=operation,
                    observed=observed.get(operation, 0.0),
                    baseline=float(expected[operation])
                    if operation in expected
                    else 0.0,
                    band=band,
                )
            )
    return tuple(deviations)


# --- Fidelity harness (VALIDATION §7.6) --------------------------------------
#
# Отдельный harness: проверяет **преобразователь** (LLM), а не хост. Абсолютные
# значения тона непроверяемы — проверяется **порядок** сигнала (монотонность),
# как аналог ``check_direction``. Три класса искажения: маскирование (сигнал не
# читается), инверсия (знак перевёрнут), фабрикация (сигнал добавлен). LLM-as-
# judge для ворот не используется (ADR-0007); семантика — HITL.


class ToneAxis(Enum):
    """Ось сигнала, которую преобразователь должен сохранить (VALIDATION §7.6).

        VALENCE — знак/порядок настроения;
        STRESS — величина напряжения;
        GOAL_SCOPE — следование цели/охвату реплики (вопрос vs утверждение).
    """

    VALENCE = "valence"
    STRESS = "stress"
    GOAL_SCOPE = "goal_scope"


class DistortionClass(Enum):
    """Класс искажения сигнала преобразователем (VALIDATION §7.6)."""

    NONE = "none"
    MASKING = "masking"  # сигнал не читается (порядок не сохранился/схлопнулся)
    INVERSION = "inversion"  # знак перевёрнут
    FABRICATION = "fabrication"  # сигнал добавлен (тон вне рамок)


@runtime_checkable
class ToneScorer(Protocol):
    """Контракт скорера тона: текст → числовая оценка по оси.

    Скорер не судит «правильность» реплики — он измеряет сигнал по оси, чтобы
    harness сравнил *порядок* двух реплик (VALIDATION §7.6).
    """

    def score(self, text: str, axis: ToneAxis) -> float:
        """Оценить текст по оси.

        Args:
            text: Текст реплики.
            axis: Ось сигнала.

        Returns:
            Числовая оценка (больше — сильнее выражен сигнал по оси).
        """
        ...


# Детерминированный лексикон полярности (baseline CI). Не «семантическая
# истина», а грубый измеритель знака/напряжения, достаточный для проверки
# порядка. Расширяется по мере появления механизмов.
_POSITIVE_LEXICON: frozenset[str] = frozenset(
    {"рад", "хорошо", "отлично", "спасибо", "приятно", "люблю", "нравится", "да"}
)
_NEGATIVE_LEXICON: frozenset[str] = frozenset(
    {"плохо", "тяжело", "устал", "больно", "грустно", "нет", "боюсь", "злюсь"}
)
_STRESS_LEXICON: frozenset[str] = frozenset(
    {"тяжело", "напряж", "устал", "больно", "стресс", "нагрузк", "срочно"}
)


@dataclass(frozen=True)
class LexiconToneScorer:
    """Детерминированный скорер тона по лексикону полярности (CI baseline).

    Считает доли слов из позитивного/негативного/стрессового лексиконов.
    Для ``GOAL_SCOPE`` возвращает 1.0 для вопроса (сигнал «уточнение») и 0.0
    иначе — структурная, а не лексическая ось.
    """

    def score(self, text: str, axis: ToneAxis) -> float:
        """Оценить текст по оси (детерминированно).

        Args:
            text: Текст реплики.
            axis: Ось сигнала.

        Returns:
            Оценка тона по оси.
        """
        low = text.lower()
        tokens = [t.strip(".,!?;:—-()") for t in low.split()]
        if axis is ToneAxis.GOAL_SCOPE:
            return 1.0 if "?" in text else 0.0
        if axis is ToneAxis.VALENCE:
            pos = sum(1 for t in tokens if t in _POSITIVE_LEXICON)
            neg = sum(1 for t in tokens if t in _NEGATIVE_LEXICON)
            return float(pos - neg)
        # STRESS: счёт стрессовых токенов + восклицания как напряжение.
        stress = sum(1 for t in tokens if t in _STRESS_LEXICON)
        stress += text.count("!")
        return float(stress)


@dataclass(frozen=True)
class EmbeddingToneScorer:
    """Opt-in скорер тона по якорным эмбеддингам (реальный ``Embedder``).

    Сравнивает эмбеддинг текста с якорными фразами оси (косинус): тон —
    близость к положительному/отрицательному/напряжённому полюсу. Используется
    только в opt-in-прогоне (сеть), не в CI (VALIDATION §7.6).

    Attributes:
        embedder: Источник эмбеддингов.
    """

    embedder: Embedder

    def score(self, text: str, axis: ToneAxis) -> float:
        """Оценить текст по оси через близость к якорям.

        Args:
            text: Текст реплики.
            axis: Ось сигнала.

        Returns:
            Оценка тона по оси (косинусная близость к полюсу).
        """
        import numpy as np

        if axis is ToneAxis.GOAL_SCOPE:
            return 1.0 if "?" in text else 0.0
        if axis is ToneAxis.VALENCE:
            positive, negative = "мне хорошо и радостно", "мне плохо и тяжело"
        else:
            positive, negative = "мне тяжело и напряжённо", "мне спокойно и легко"
        vec = self.embedder.embed(text)
        pos = self.embedder.embed(positive)
        neg = self.embedder.embed(negative)
        return float(np.dot(vec, pos) - np.dot(vec, neg))


@dataclass(frozen=True)
class FidelityPair:
    """Пара frame'ов с ожидаемым порядком по оси (VALIDATION §7.6).

    Проверяется порядок, не абсолют: если сигнал ``frame_a`` слабее ``frame_b``
    по оси, то и тон реплики ``a`` не должен быть выше тона ``b`` (монотонность).
    ``expect_ordered=False`` — контроль на фабрикацию: сигнала быть не должно
    (frame'ы нейтральны по оси), значит тона обязаны совпасть.

    Attributes:
        id: Идентификатор пары.
        frame_a: Первый frame (слабее по оси, если ``expect_ordered``).
        frame_b: Второй frame (сильнее по оси).
        axis: Проверяемая ось.
        expect_ordered: Ожидается ли монотонность ``tone(a) ≤ tone(b)``.
    """

    id: str
    frame_a: IntentFrame
    frame_b: IntentFrame
    axis: ToneAxis
    expect_ordered: bool = True


@dataclass(frozen=True)
class FidelityResult:
    """Итог проверки одной пары.

    Attributes:
        pair: Проверенная пара.
        score_a: Тон реплики ``a`` по оси.
        score_b: Тон реплики ``b`` по оси.
        ordered: Выполнена ли монотонность ``score_a ≤ score_b``.
        distortion: Классифицированное искажение.
        passed: Выполнено ли ожидание (ordered == expect_ordered).
        reason: Причина вердикта.
    """

    pair: FidelityPair
    score_a: float
    score_b: float
    ordered: bool
    distortion: DistortionClass
    passed: bool
    reason: str


def check_fidelity(
    pair: FidelityPair,
    *,
    score_a: float,
    score_b: float,
    tol: float = 1e-9,
) -> FidelityResult:
    """Проверить сохранение сигнала (чистая, VALIDATION §7.6).

    Семантика ``pair.expect_ordered``:

    * ``True`` — сигнал **ожидается** и упорядочен ``tone(a) ≤ tone(b)``.
      Искажения: ``INVERSION`` (порядок перевёрнут), ``MASKING`` (тона
      схлопнулись — сигнал не читается).
    * ``False`` — сигнала **быть не должно** (frame'ы нейтральны по оси).
      Искажение: ``FABRICATION`` (тон добавлен там, где сигнала нет).

    Args:
        pair: Пара frame'ов.
        score_a: Тон реплики ``a``.
        score_b: Тон реплики ``b``.
        tol: Допуск на схлопывание/равенство.

    Returns:
        :class:`FidelityResult`.
    """
    if not all(math.isfinite(s) for s in (score_a, score_b)):
        return FidelityResult(
            pair=pair,
            score_a=score_a,
            score_b=score_b,
            ordered=False,
            distortion=DistortionClass.MASKING,
            passed=False,
            reason="non-finite tone",
        )

    collapsed = abs(score_a - score_b) <= tol
    ordered = score_a <= score_b + tol

    if not pair.expect_ordered:
        # Сигнала быть не должно: добавленный тон — фабрикация.
        passed = collapsed
        distortion = DistortionClass.NONE if passed else DistortionClass.FABRICATION
        reason = "ok" if passed else f"{pair.axis.value}: fabricated signal"
    elif collapsed:
        passed = False
        distortion = DistortionClass.MASKING
        reason = f"{pair.axis.value}: signal masked"
    elif not ordered:
        passed = False
        distortion = DistortionClass.INVERSION
        reason = f"{pair.axis.value}: order inverted"
    else:
        passed = True
        distortion = DistortionClass.NONE
        reason = "ok"

    return FidelityResult(
        pair=pair,
        score_a=score_a,
        score_b=score_b,
        ordered=ordered,
        distortion=distortion,
        passed=passed,
        reason=reason,
    )


def default_fidelity_pairs() -> tuple[FidelityPair, ...]:
    """Встроенный корпус пар верности (минимум осей: valence/stress/scope).

    Returns:
        Кортеж :class:`FidelityPair`.
    """
    return (
        FidelityPair(
            id="fidelity.valence.order",
            frame_a=build_intent_frame(
                f=1.0, valence=-3.0, stress=0.0, task="none", goal="respond"
            ),
            frame_b=build_intent_frame(
                f=1.0, valence=3.0, stress=0.0, task="none", goal="respond"
            ),
            axis=ToneAxis.VALENCE,
        ),
        FidelityPair(
            id="fidelity.stress.order",
            frame_a=build_intent_frame(
                f=1.0, valence=0.0, stress=0.0, task="none", goal="respond"
            ),
            frame_b=build_intent_frame(
                f=1.0, valence=0.0, stress=9.0, task="none", goal="respond"
            ),
            axis=ToneAxis.STRESS,
        ),
        FidelityPair(
            id="fidelity.scope.order",
            frame_a=build_intent_frame(
                f=1.0, valence=0.0, stress=0.0, task="none", goal="respond"
            ),
            frame_b=build_intent_frame(
                f=1.0, valence=0.0, stress=0.0, task="none", goal="explore"
            ),
            axis=ToneAxis.GOAL_SCOPE,
        ),
        # Контроль на фабрикацию: оба frame'а нейтральны по valence — тон не
        # должен «придумываться».
        FidelityPair(
            id="fidelity.valence.no_fabrication",
            frame_a=build_intent_frame(
                f=1.0, valence=0.0, stress=0.0, task="none", goal="respond"
            ),
            frame_b=build_intent_frame(
                f=1.0, valence=0.0, stress=0.0, task="none", goal="respond"
            ),
            axis=ToneAxis.VALENCE,
            expect_ordered=False,
        ),
    )


class FidelityHarness:
    """Shell: прогон пар верности через инъектируемый responder (VALIDATION §7.6).

    Responder — ``IntentFrame → str``. В CI это детерминированная функция
    (в т.ч. намеренно искажающая — для проверки детекторов); в opt-in-прогоне —
    реальная LLM (обёртка ``SpeechController``). Скорер — ``LexiconToneScorer``
    по умолчанию; ``EmbeddingToneScorer`` — opt-in.

    Attributes:
        responder: Преобразователь frame → текст реплики.
        scorer: Скорер тона.
    """

    def __init__(
        self,
        responder: Callable[[IntentFrame], str],
        *,
        scorer: ToneScorer | None = None,
    ) -> None:
        self.responder = responder
        self.scorer: ToneScorer = (
            scorer if scorer is not None else LexiconToneScorer()
        )

    def run(self, pair: FidelityPair) -> FidelityResult:
        """Прогнать одну пару и проверить порядок.

        Args:
            pair: Пара frame'ов.

        Returns:
            :class:`FidelityResult`.
        """
        text_a = self.responder(pair.frame_a)
        text_b = self.responder(pair.frame_b)
        score_a = self.scorer.score(text_a, pair.axis)
        score_b = self.scorer.score(text_b, pair.axis)
        return check_fidelity(pair, score_a=score_a, score_b=score_b)

    def run_all(
        self, pairs: Sequence[FidelityPair] | None = None
    ) -> list[FidelityResult]:
        """Прогнать корпус пар (по умолчанию — встроенный).

        Args:
            pairs: Пары (None → :func:`default_fidelity_pairs`).

        Returns:
            Список :class:`FidelityResult` в порядке пар.
        """
        corpus = default_fidelity_pairs() if pairs is None else pairs
        return [self.run(pair) for pair in corpus]


def llm_responder(llm: Any) -> Callable[[IntentFrame], str]:
    """Обернуть реальный LLM в responder ``frame → текст`` (opt-in).

    Строит chat-сообщения из frame (как ``SpeechController``) и вызывает LLM.
    Используется в opt-in-прогоне fidelity с настоящей моделью
    (VALIDATION §7.6); в CI не задействуется (сеть, ADR-0007).

    Args:
        llm: LLM-клиент (``ApiLlmClient`` и т.п.).

    Returns:
        Callable ``IntentFrame → str``.
    """

    def responder(frame: IntentFrame) -> str:
        from src.speech.intent import register_max_tokens, render_messages

        messages = render_messages(frame, user_text="")
        return llm.reply(messages, max_tokens=register_max_tokens(frame.register))

    return responder
