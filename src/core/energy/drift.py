"""Drift detector — заготовка самоконтроля (манифест §6.4).

S1: только вычисляет флаг «состояние вышло за границы дольше hold_ticks» и
отдаёт его в телеметрию. Реакция (протокол сброса) — S6.

Functional Core / Imperative Shell (ADR-0004): детектор — Shell (хранит
счётчик удержания), логика проверки границ — тривиальна и чиста.
"""

from __future__ import annotations

from .models import FreeEnergyResult


class DriftDetector:
    """Детектор тихого дрейфа состояния.

    Считает тики подряд, на которых F(t) или stress превышают границы.
    Если превышение держится дольше ``hold_ticks`` — флаг drift = True.

    Attributes:
        f_threshold: Порог F(t).
        stress_threshold: Порог allostatic_stress.
        hold_ticks: Сколько тиков подряд нужно превышение.
    """

    def __init__(
        self,
        f_threshold: float,
        stress_threshold: float,
        hold_ticks: int = 20,
    ) -> None:
        """Создать детектор дрейфа.

        Args:
            f_threshold: Порог F(t).
            stress_threshold: Порог стресса.
            hold_ticks: Минимальная длительность превышения, тики.

        Raises:
            ValueError: Если hold_ticks < 1 или пороги < 0.
        """
        if hold_ticks < 1:
            raise ValueError(f"hold_ticks must be >= 1, got {hold_ticks}")
        if f_threshold < 0.0 or stress_threshold < 0.0:
            raise ValueError("thresholds must be >= 0")

        self.f_threshold = f_threshold
        self.stress_threshold = stress_threshold
        self.hold_ticks = hold_ticks
        self._streak: int = 0

    def update(self, result: FreeEnergyResult) -> bool:
        """Обновить детектор результатом тика.

        Args:
            result: Результат текущего тика.

        Returns:
            True, если превышение границ держится дольше hold_ticks.
        """
        exceeded = (
            result.f > self.f_threshold
            or result.allostatic_stress > self.stress_threshold
        )
        if exceeded:
            self._streak += 1
        else:
            self._streak = 0
        return self._streak > self.hold_ticks

    @property
    def streak(self) -> int:
        """Текущая длина серии превышений (тики)."""
        return self._streak

    def reset(self) -> None:
        """Сбросить счётчик серии."""
        self._streak = 0
