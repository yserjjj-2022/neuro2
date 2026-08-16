import logging
from typing import Callable, Optional

import numpy as np

from .calculator import FreeEnergyCalculator
from .models import FreeEnergyResult

logger = logging.getLogger(__name__)


class EnergyObserver:
    """Shadow-наблюдатель: логирует F(t) без принятия решений.

    Functional Core / Imperative Shell:
    - Core (calculator) — чистая функция, тестируется без I/O
    - Shell (observer) — инъекция sink, можно мокать в тестах
    - В проде: sink=lambda r: telemetry_logger.log(r.f, r.valence, r.allostatic_stress)
      (active_columns=0 по умолчанию до появления src/core/cmc/ в Фазе 3)
    - В тестах: sink = list.append
    """

    def __init__(
        self,
        calculator: FreeEnergyCalculator,
        sink: Optional[Callable[[FreeEnergyResult], None]] = None,
    ) -> None:
        self.calculator = calculator
        self.sink = sink
        self._prev_f: float = 0.0
        self._prev_stress: float = 0.0

    def observe(
        self,
        prediction_error: np.ndarray,
        precision: np.ndarray,
    ) -> FreeEnergyResult:
        """Наблюдать за состоянием: считать метрики, записать через sink.

        Args:
            prediction_error: Вектор ошибки предсказания e(t).
            precision: Вектор точности γ.

        Returns:
            FreeEnergyResult — сырые метрики без принятия решений.
        """
        result = self.calculator.compute(
            prediction_error, precision, self._prev_f, self._prev_stress
        )
        self._prev_f = result.f
        self._prev_stress = result.allostatic_stress

        if self.sink is not None:
            self.sink(result)

        return result
