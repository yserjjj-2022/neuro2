"""Domain objects for policy module.

Frozen dataclasses + enum analogous to FreeEnergyResult (energy) and
TaskAttraction (attractors): immutable snapshots. ``PolicyContext`` is the
*extensible* input — S6 adds metacognitive observables as a new field with a
default, without breaking callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from src.core.homeostasis import HomeostasisState


@runtime_checkable
class PartnerView(Protocol):
    """Структурный вид состояния партнёра для policy (S5).

    Policy не импортирует слой ``tm`` напрямую (без цикла зависимостей):
    достаточно, чтобы объект предоставлял эти read-only атрибуты.
    ``tm.PartnerState`` (frozen dataclass) структурно удовлетворяет контракту.
    """

    @property
    def trust(self) -> float: ...

    @property
    def ambiguity(self) -> float: ...

    @property
    def conflict(self) -> float: ...

    @property
    def uncertainty(self) -> float: ...

    @property
    def name(self) -> str: ...


@runtime_checkable
class MetacognitionView(Protocol):
    """Структурный вид метакогнитивных наблюдаемых для policy (S6).

    Policy не импортирует ``core.selfcontrol`` напрямую (без цикла): достаточно
    read-only атрибутов. ``selfcontrol.Metacognition`` структурно удовлетворяет
    контракту. ``None`` → S5-совместимость (эпистемический драйв выключен).
    """

    @property
    def conflict(self) -> float: ...

    @property
    def metastability(self) -> float: ...

    @property
    def epistemic_uncertainty(self) -> float: ...

    @property
    def saturation(self) -> float: ...


class Action(Enum):
    """Кандидаты-действия policy.

    На S4 — только речевое поведение (внешние side-effect отложены, решение B
    из stages/S4_SPEC.md).
    """

    RESPOND = "respond"
    SILENT = "silent"
    INITIATIVE = "initiative"
    IDENTIFY_PARTNER = "identify_partner"
    EXPLORE = "explore"  # S6: эпистемический драйв (исследование неопределённости)


@dataclass(frozen=True)
class Preferences:
    """Предпочитаемые исходы (goal-directed: меняются без переобучения).

    Attributes:
        respond_to_messages: Предпочитаем ли отвечать на сообщение.
        initiative_f_threshold: Порог F для инициативы без сообщения.
        homeostatic_alert: Предпочитаем ли предупреждать о перегрузке.
        alert_deviation: Порог отклонения гомеостаза для тревоги (предупредить).
        silent_baseline: Базовая ценность SILENT (безопасный дефолт).
        silent_stress_gain: Прирост ценности SILENT со стрессом (беречь ресурс).
        pragmatic_weight: Вес прагматической ценности.
        epistemic_weight: Вес эпистемической ценности.
    """

    respond_to_messages: bool = True
    initiative_f_threshold: float = 1.0
    homeostatic_alert: bool = True
    alert_deviation: float = 0.7
    silent_baseline: float = 0.5
    silent_stress_gain: float = 0.3
    pragmatic_weight: float = 1.0
    epistemic_weight: float = 0.5
    identify_threshold: float = 0.7  # S5: порог uncertainty для мягкого интента
    partner_trust_floor: float = 0.5  # S5: нижняя граница масштаба RESPOND
    explore_threshold: float = 0.6  # S6: порог неопределённости для EXPLORE

    def __post_init__(self) -> None:
        """Валидация: неотрицательные пороги и веса.

        Raises:
            ValueError: Если initiative_f_threshold < 0, alert_deviation вне
                [0, 1], silent_baseline/silent_stress_gain/pragmatic_weight/
                epistemic_weight < 0.
        """
        if self.initiative_f_threshold < 0.0:
            raise ValueError(
                f"initiative_f_threshold must be >= 0, "
                f"got {self.initiative_f_threshold}"
            )
        if not 0.0 <= self.alert_deviation <= 1.0:
            raise ValueError(
                f"alert_deviation must be in [0, 1], got {self.alert_deviation}"
            )
        if self.silent_baseline < 0.0:
            raise ValueError(
                f"silent_baseline must be >= 0, got {self.silent_baseline}"
            )
        if self.silent_stress_gain < 0.0:
            raise ValueError(
                f"silent_stress_gain must be >= 0, got {self.silent_stress_gain}"
            )
        if self.pragmatic_weight < 0.0:
            raise ValueError(
                f"pragmatic_weight must be >= 0, got {self.pragmatic_weight}"
            )
        if self.epistemic_weight < 0.0:
            raise ValueError(
                f"epistemic_weight must be >= 0, got {self.epistemic_weight}"
            )
        if not 0.0 <= self.identify_threshold <= 1.0:
            raise ValueError(
                f"identify_threshold must be in [0, 1], got {self.identify_threshold}"
            )
        if not 0.0 <= self.partner_trust_floor <= 1.0:
            raise ValueError(
                f"partner_trust_floor must be in [0, 1], "
                f"got {self.partner_trust_floor}"
            )
        if not 0.0 <= self.explore_threshold <= 1.0:
            raise ValueError(
                f"explore_threshold must be in [0, 1], got {self.explore_threshold}"
            )


@dataclass(frozen=True)
class PolicyContext:
    """Расширяемый вход policy.

    S6 добавляет метакогнитивные наблюдаемые новым полем с дефолтом — это
    условие безболезненного переноса метакогниции (решение A, S4_SPEC).

    Attributes:
        f: Свободная энергия F(t).
        valence: Валентность.
        stress: Аллостатический стресс.
        task: Активная задача/аттрактор (тег колонки).
        homeostasis: Снимок гомеостаза.
        has_new_message: Пришло ли новое сообщение оператора.
        mode: Режим хоста (game/cooperative/free; макро-контекст).
    """

    f: float
    valence: float
    stress: float
    task: str
    homeostasis: HomeostasisState
    has_new_message: bool
    mode: str = "free"
    partner: PartnerView | None = None  # S5: ToM; None → S4-совместимость
    metacognition: MetacognitionView | None = None  # S6; None → S5-совместимость


@dataclass(frozen=True)
class MacroContext:
    """Минимальный макро-контекст (дискретный слой без pymdp, S4).

    Фиксирует активную задачу и режим хоста как единый источник для policy.
    Полная sparse-факторизация (режим/партнёр/задача) — S6 (ADR-0005 §3).

    Attributes:
        task: Активная задача/аттрактор (тег колонки).
        mode: Режим хоста (game/cooperative/free, манифест §8).
    """

    task: str = "none"
    mode: str = "free"

    def __post_init__(self) -> None:
        """Валидация режима хоста.

        Raises:
            ValueError: Если mode не из {game, cooperative, free}.
        """
        if self.mode not in ("game", "cooperative", "free"):
            raise ValueError(f"mode must be game|cooperative|free, got {self.mode!r}")


@dataclass(frozen=True)
class PolicyCandidate:
    """Оценка одного кандидата-действия.

    Attributes:
        action: Действие.
        pragmatic: Прагматическая ценность (близость к предпочитаемому исходу).
        epistemic: Эпистемическая ценность (снижение неопределённости).
        value: Итоговая ценность (взвешенная сумма).
        reason: Причина оценки (для трассировки).
    """

    action: Action
    pragmatic: float
    epistemic: float
    value: float
    reason: str


@dataclass(frozen=True)
class PolicyTrace:
    """Причинная трассировка решения policy (explainability — инвариант).

    Attributes:
        chosen: Выбранное действие.
        reason: Причина выбора, выведенная из расчёта (не постфактум-текст).
        candidates: Все рассмотренные кандидаты с оценками.
    """

    chosen: Action
    reason: str
    candidates: tuple[PolicyCandidate, ...]
