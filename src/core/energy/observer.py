import logging
from collections.abc import Callable

import numpy as np

from .calculator import FreeEnergyCalculator
from .models import EnergyState, FreeEnergyResult

logger = logging.getLogger(__name__)


class EnergyObserver:
    """Shadow-наблюдатель: логирует F(t) без принятия решений.

    Functional Core / Imperative Shell:
    - Core (calculator) — чистая функция, тестируется без I/O
    - Shell (observer) — владеет EnergyState, инъекция sink, мокается в тестах
    - В проде: sink=lambda r: telemetry_logger.log(...)
    - В тестах: sink = list.append
    """

    def __init__(
        self,
        calculator: FreeEnergyCalculator,
        sink: Callable[[FreeEnergyResult], None] | None = None,
    ) -> None:
        self.calculator = calculator
        self.sink = sink
        self._state: EnergyState = EnergyState()

    def observe(
        self,
        prediction_error: np.ndarray,
        precision: np.ndarray,
        dt: float,
    ) -> FreeEnergyResult:
        """Наблюдать за состоянием: считать метрики, записать через sink.

        Args:
            prediction_error: Вектор ошибки предсказания e(t).
            precision: Вектор точности γ.
            dt: Шаг интегрирования в секундах (> 0).

        Returns:
            FreeEnergyResult — сырые метрики без принятия решений.
        """
        result = self.calculator.compute(prediction_error, precision, self._state, dt)
        self._state = EnergyState(
            f=result.f,
            stress=result.allostatic_stress,
            valence=result.valence,
        )

        if self.sink is not None:
            self.sink(result)

        return result

    @property
    def state(self) -> EnergyState:
        """Текущее состояние аффективного контура."""
        return self._state
