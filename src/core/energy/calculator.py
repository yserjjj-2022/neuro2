import logging

import numpy as np

from .models import EnergyState, FreeEnergyResult

logger = logging.getLogger(__name__)


class FreeEnergyCalculator:
    """Чистый калькулятор свободной энергии — полностью stateless.

    Соответствует паттерну Functional Core:
    - Все вычисления — чистые функции без побочных эффектов
    - Состояние (EnergyState) передаётся явно
    - Один и тот же input → один и тот же output

    Все временные величины — в секундах (S1: единая временная база).
    ``dt`` передаётся на каждый вызов, калькулятор не хранит шаг.
    """

    def __init__(
        self,
        stress_leak_per_sec: float = 1.0,
        valence_tau: float = 0.1,
        gamma_base: float = 1.0,
    ) -> None:
        """Инициализация калькулятора.

        Args:
            stress_leak_per_sec: λ — скорость утечки аллостатического
                стресса, 1/с. Маппинг старого ``stress_decay=0.99`` на тик
                при 100 Гц: ``λ ≈ 1.0`` 1/с (half-life ≈ 0.69 с).
            valence_tau: τ — постоянная времени сглаживания валентности, с.
                Меньше → быстрее реакция, больше → сильнее сглаживание.
            gamma_base: γ по умолчанию для пустого precision.
        """
        self.stress_leak_per_sec = stress_leak_per_sec
        self.valence_tau = valence_tau
        self.gamma_base = gamma_base

    def compute(
        self,
        prediction_error: np.ndarray,
        precision: np.ndarray,
        state: EnergyState,
        dt: float,
        importance: np.ndarray | None = None,
    ) -> FreeEnergyResult:
        """Рассчитать F(t), valence, stress, gamma.

        Чистая функция: не изменяет внутреннее состояние.

        Args:
            prediction_error: Вектор ошибки предсказания e(t).
            precision: Вектор точности γ для каждого канала.
            state: Предыдущее состояние аффективного контура.
            dt: Шаг интегрирования в секундах (> 0).
            importance: Веса важности каналов wᵢ (по компонентам), shape как
                prediction_error. None → единицы (обратная совместимость:
                F(t) = 0.5·Σγᵢeᵢ²). Ненулевой вектор чинит скрытый вес
                размерности (см. BACKLOG «Честная обработка сигналов»).

        Returns:
            FreeEnergyResult с полями: f, valence, allostatic_stress, gamma.

        Raises:
            ValueError: Если prediction_error.shape != precision.shape (или
                importance), или dt <= 0.

        Formula:
            F(t) = 0.5 · Σᵢ γᵢ · wᵢ · e(t)ᵢ²
            valence_raw = -(F(t) - state.f) / dt
            a = 1 - exp(-dt / valence_tau)
            valence = (1 - a)·state.valence + a·valence_raw
            stress = state.stress · exp(-λ·dt) + F(t)·dt
            gamma = mean(precision) if len(precision) > 0 else gamma_base

        Note:
            gamma = mean(precision) — простейшая агрегация для Фазы 1.
            Пересмотр (min, geometric mean) — Фаза 2.
            precision <= 0 клиппится до 1e-6.
        """
        # 1. Validate dt (fail-fast): единая временная база S1
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")

        # 2. Validate shapes (fail-fast) — catches empty vs non-empty too
        if prediction_error.shape != precision.shape:
            raise ValueError(
                f"Shape mismatch: prediction_error {prediction_error.shape} "
                f"!= precision {precision.shape}"
            )
        if importance is not None and importance.shape != prediction_error.shape:
            raise ValueError(
                f"Shape mismatch: importance {importance.shape} "
                f"!= prediction_error {prediction_error.shape}"
            )

        # 3. Handle empty arrays (if shape check passed, both are empty)
        if prediction_error.size == 0:
            valence_raw = -(0.0 - state.f) / dt
            a = 1.0 - float(np.exp(-dt / self.valence_tau))
            return FreeEnergyResult(
                f=0.0,
                valence=(1.0 - a) * state.valence + a * valence_raw,
                allostatic_stress=state.stress
                * float(np.exp(-self.stress_leak_per_sec * dt)),
                gamma=self.gamma_base,
            )

        # 4. Clip precision (silent clip for Phase 1)
        if np.any(precision <= 0):
            logger.debug("precision clipped")
            precision = np.clip(precision, 1e-6, None)

        # 5. Compute F(t) — с весами важности, если заданы
        error_sq = prediction_error**2
        if importance is None:
            f = float(0.5 * np.sum(precision * error_sq))
        else:
            f = float(0.5 * np.sum(precision * importance * error_sq))

        # 6. Valence: raw derivative, exponentially smoothed over time
        valence_raw = -(f - state.f) / dt
        a = 1.0 - float(np.exp(-dt / self.valence_tau))
        valence = (1.0 - a) * state.valence + a * valence_raw

        # 7. Stress: exponential leak + accumulation in seconds
        stress = state.stress * float(np.exp(-self.stress_leak_per_sec * dt)) + f * dt

        # 8. Gamma: mean precision
        gamma = float(np.mean(precision))

        return FreeEnergyResult(
            f=f,
            valence=valence,
            allostatic_stress=stress,
            gamma=gamma,
        )
