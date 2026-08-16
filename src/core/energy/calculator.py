import logging

import numpy as np

from .models import FreeEnergyResult

logger = logging.getLogger(__name__)


class FreeEnergyCalculator:
    """Чистый калькулятор свободной энергии — полностью stateless.

    Соответствует паттерну Functional Core:
    - Все вычисления — чистые функции без побочных эффектов
    - Состояние (prev_f, prev_stress) хранится вызывающим кодом
    - Один и тот же input → один и тот же output без зависимости от порядка вызовов
    """

    def __init__(
        self,
        dt: float = 0.01,
        stress_decay: float = 0.99,
        gamma_base: float = 1.0,
    ) -> None:
        self.dt = dt
        self.stress_decay = stress_decay
        self.gamma_base = gamma_base

    def compute(
        self,
        prediction_error: np.ndarray,
        precision: np.ndarray,
        prev_f: float,
        prev_stress: float,
    ) -> FreeEnergyResult:
        """Рассчитать F(t), valence, stress, gamma.

        Чистая функция: не изменяет внутреннее состояние.

        Args:
            prediction_error: Вектор ошибки предсказания e(t).
            precision: Вектор точности γ для каждого канала.
            prev_f: Значение F(t-1).
            prev_stress: Значение allostatic_stress(t-1).

        Returns:
            FreeEnergyResult с полями: f, valence, stress, gamma.

        Raises:
            ValueError: Если prediction_error.shape != precision.shape.

        Formula:
            F(t) = 0.5 · Σᵢ γᵢ · e(t)ᵢ²
            valence = -(F(t) - prev_f) / dt
            stress = prev_stress * stress_decay + F(t)
            gamma = mean(precision) if len(precision) > 0 else gamma_base

        Note:
            gamma = mean(precision) — простейшая агрегация для Фазы 1.
            Пересмотр (min, geometric mean) — Фаза 2.
            precision <= 0 клиппится до 1e-6.
        """
        # 1. Validate shapes (fail-fast) — catches empty vs non-empty too
        if prediction_error.shape != precision.shape:
            raise ValueError(
                f"Shape mismatch: prediction_error {prediction_error.shape} "
                f"!= precision {precision.shape}"
            )

        # 2. Handle empty arrays (if shape check passed, both are empty)
        if prediction_error.size == 0:
            return FreeEnergyResult(
                f=0.0,
                valence=-(0.0 - prev_f) / self.dt,
                allostatic_stress=prev_stress * self.stress_decay,
                gamma=self.gamma_base,
            )

        # 3. Clip precision (silent clip for Phase 1)
        if np.any(precision <= 0):
            logger.debug("precision clipped")
            precision = np.clip(precision, 1e-6, None)

        # 4. Compute F(t)
        f = 0.5 * np.sum(precision * prediction_error**2)

        # 5. Compute valence, stress, gamma
        valence = -(f - prev_f) / self.dt
        stress = prev_stress * self.stress_decay + f
        gamma = float(np.mean(precision))

        return FreeEnergyResult(
            f=f,
            valence=valence,
            allostatic_stress=stress,
            gamma=gamma,
        )
