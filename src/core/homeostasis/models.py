"""Domain objects for homeostasis module.

Frozen dataclasses analogous to FreeEnergyResult (energy), TaskAttraction
(attractors) and SignalSource (mcp): immutable snapshots of the homeostatic
evaluation. No I/O, no decisions — the module reports deviation, the shell
(policy/reflex) decides.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Setpoint:
    """Сетпоинт интероцептивного канала.

    Задаёт целевой диапазон канала: до ``comfort`` — норма, при ``critical``
    отклонение максимально. ``critical`` согласован с порогом reflex-пути
    (``severity >= 0.9``), поэтому максимальное отклонение и критический
    сигнал наступают одновременно.

    Attributes:
        tag: Тег источника ("battery", "resources", "cpu").
        comfort: Severity, до которой канал считается в норме, [0, 1].
        critical: Severity, при которой отклонение = 1.0, (comfort, 1].
        weight: Важность канала (корневой приор; S4 — дефолт 1.0).
    """

    tag: str
    comfort: float = 0.5
    critical: float = 0.9
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Валидация границ сетпоинта (fail-fast).

        Raises:
            ValueError: Если tag пуст, comfort/critical вне [0, 1],
                comfort >= critical или weight <= 0.
        """
        if not self.tag:
            raise ValueError("tag must not be empty")
        if not 0.0 <= self.comfort <= 1.0:
            raise ValueError(f"comfort must be in [0, 1], got {self.comfort}")
        if not 0.0 <= self.critical <= 1.0:
            raise ValueError(f"critical must be in [0, 1], got {self.critical}")
        if self.comfort >= self.critical:
            raise ValueError(
                f"comfort must be < critical, got {self.comfort} >= {self.critical}"
            )
        if self.weight <= 0.0:
            raise ValueError(f"weight must be > 0, got {self.weight}")


@dataclass(frozen=True)
class HomeostaticSignal:
    """Оценка одного канала относительно сетпоинта.

    Attributes:
        tag: Тег канала.
        severity: Сырая severity сигнала (из ``SignalSource``), [0, 1].
        deviation: Нормированное отклонение, [0, 1].
        is_critical: ``severity >= reflex_threshold`` (критический сигнал).
    """

    tag: str
    severity: float
    deviation: float
    is_critical: bool


@dataclass(frozen=True)
class HomeostasisState:
    """Снимок гомеостаза после оценки всех каналов.

    Attributes:
        signals: Оценки по каналам (в порядке сетпоинтов).
        max_deviation: Максимум ``deviation`` — вход policy.
        severity: Максимум ``severity`` — вход reflex-пути.
        is_critical: Есть ли хотя бы один критический канал.
    """

    signals: tuple[HomeostaticSignal, ...]
    max_deviation: float
    severity: float
    is_critical: bool
