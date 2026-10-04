"""Domain objects for selfcontrol module (S6).

Frozen dataclasses and enums analogous to FreeEnergyResult (energy) and
TaskAttraction (attractors). ``Metacognition`` is the read-only snapshot fed to
policy (ADR-0009 §1); the reset protocol is expressed by ``ResetPlan`` /
``ChangeAssessment`` (ADR-0009 §2–3).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Metacognition:
    """Снимок метакогнитивных наблюдаемых (read-only, не дублирует аффект).

    Считается из уже существующих данных (scores, история маски/переключений,
    тренд F, неопределённость партнёра). Подаётся в policy как отдельное поле
    контекста; в F того же тика не влияет (ADR-0009 §1).

    Attributes:
        conflict: Несогласие ансамбля (разброс активностей), [0, 1].
        metastability: Частота смен аттрактора в окне, [0, 1].
        epistemic_uncertainty: Неопределённость идентичности/предсказания, [0, 1].
        saturation: Насыщение/тренд F (хронизация нагрузки), [0, 1].
    """

    conflict: float
    metastability: float
    epistemic_uncertainty: float
    saturation: float

    def __post_init__(self) -> None:
        for name in (
            "conflict",
            "metastability",
            "epistemic_uncertainty",
            "saturation",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class CriticalSlowingDown:
    """Признак приближения к смене режима (Scheffer/Dakos, ADR-0009 §2).

    Attributes:
        variance: Дисперсия метрики в окне.
        autocorrelation: Lag-1 автокорреляция метрики в окне.
        slowing: Нормированный признак slowing down, [0, 1].
        is_warning: slowing >= порога (триггер-кандидат).
    """

    variance: float
    autocorrelation: float
    slowing: float
    is_warning: bool


class ChangeKind(Enum):
    """Классификация изменения (INTENT §4, ADR-0009 §3)."""

    STABLE = "stable"
    DEVELOPMENT = "development"
    DRIFT = "drift"


@dataclass(frozen=True)
class ChangeAssessment:
    """Оценка изменения по трём осям INTENT §4.

    Attributes:
        kind: stable / development / drift.
        core_preserved: Ядро сохранено (не нарушено).
        traceable: Изменение объяснимо опытом.
        coherent: Траектория связная (не рваная/зафиксированная).
        reason: Причина классификации (для трассировки).
    """

    kind: ChangeKind
    core_preserved: bool
    traceable: bool
    coherent: bool
    reason: str


class ResetLevel(Enum):
    """Уровень протокола сброса (три механизма, ADR-0009 §2).

    Attributes:
        SOFT: Adaptive reset — дешёвый сброс к baseline для поиска.
        FREEZE: Regime shift — управляемый переход установок (громкий).
        HARD: Catastrophic drift — аварийная остановка (никогда не «к норме»).
    """

    SOFT = "soft"
    FREEZE = "freeze"
    HARD = "hard"


@dataclass(frozen=True)
class ResetPlan:
    """План сброса (чистый, до исполнения).

    Attributes:
        level: Уровень сброса.
        reason: Причина (триггер + классификация).
        triggered: Нужно ли исполнять сброс.
    """

    level: ResetLevel
    reason: str
    triggered: bool
