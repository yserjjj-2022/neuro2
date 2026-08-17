"""Unit tests for attractors Imperative Shell (TaskAttractor).

Tests TaskAttractor stateful manager with all transition paths:
- first tick, stable input, switch, convergence, immediate_switch,
  basin_stability, reset, tick_after_reset, current_mask_none.
"""

import numpy as np
import pytest

from src.core.attractors import TaskAttractor


class TestTaskAttractor:
    """Tests for TaskAttractor Shell."""

    def test_init_n_tasks_lt_2(self) -> None:
        """n_tasks < 2 → ValueError."""
        with pytest.raises(ValueError, match="n_tasks < 2"):
            TaskAttractor(n_tasks=1)

    def test_init_valid(self) -> None:
        """n_tasks >= 2 создаётся без ошибок."""
        a = TaskAttractor(n_tasks=3)
        assert a.history_size == 0

    def test_init_custom_params(self) -> None:
        """Кастомные параметры инициализации."""
        a = TaskAttractor(
            n_tasks=3,
            base_dwell=10,
            dwell_slope=1.5,
            plasticity_gain=0.2,
            basin_threshold=0.1,
            convergence_threshold=1e-6,
        )
        assert a.history_size == 0

    def test_tick_first(self) -> None:
        """Первый тик: mask = one-hot, history = 0."""
        a = TaskAttractor(n_tasks=3)
        scores = np.array([0.2, 0.7, 0.1])
        result = a.tick(scores)
        assert result.history_size == 0
        assert result.mask[1] == 1.0
        assert result.mask[0] == 0.0
        assert result.mask[2] == 0.0
        assert result.converged is False

    def test_tick_stable(self) -> None:
        """Stable input: mask не меняется, history растёт."""
        a = TaskAttractor(n_tasks=3, base_dwell=2, plasticity_gain=0.0)
        scores = np.array([0.2, 0.7, 0.1])

        result1 = a.tick(scores)
        assert result1.history_size == 0
        assert a.history_size == 0

        result2 = a.tick(scores)
        assert result2.history_size == 1
        assert a.history_size == 1

        result3 = a.tick(scores)
        assert result3.history_size == 2
        assert a.history_size == 2

    def test_tick_switch(self) -> None:
        """Switch: новый mask, history = 0."""
        a = TaskAttractor(
            n_tasks=3,
            base_dwell=1,
            dwell_slope=0.0,
            plasticity_gain=0.0,
            basin_threshold=0.0,
        )
        # Первый тик: колонка 0 выигрывает
        scores1 = np.array([0.7, 0.2, 0.1])
        r1 = a.tick(scores1)
        assert r1.mask[0] == 1.0
        assert r1.history_size == 0

        # Второй тик: колонка 1 выигрывает явно → immediate_switch
        scores2 = np.array([0.1, 0.8, 0.1])
        r2 = a.tick(scores2)
        assert r2.mask[1] == 1.0
        assert r2.history_size == 0

    def test_tick_convergence(self) -> None:
        """Scores < τ, switch запрещён до history >= min_dwell."""
        a = TaskAttractor(
            n_tasks=3,
            base_dwell=5,
            dwell_slope=0.0,
            plasticity_gain=0.0,
            basin_threshold=0.0,
            convergence_threshold=1e-8,
        )
        # Стабильный вход → сходимость
        scores = np.array([1e-9, 0.5, 1e-9])
        for _ in range(10):
            r = a.tick(scores)

        # При сходимости: scores[0] < τ, но scores[1] = 0.5 > τ
        # converged должен быть False (max > τ)
        assert r.converged is False

    def test_tick_immediate_switch(self) -> None:
        """Явное превосходство конкурента → switch."""
        a = TaskAttractor(
            n_tasks=3,
            base_dwell=10,  # большой dwell
            dwell_slope=0.0,
            plasticity_gain=0.0,
            basin_threshold=0.0,
        )
        # Первый тик: колонка 0
        scores1 = np.array([0.5, 0.2, 0.1])
        r1 = a.tick(scores1)
        assert r1.mask[0] == 1.0

        # Второй тик: колонка 1 явно выигрывает (Δ = 0.4 > 0.3)
        scores2 = np.array([0.1, 0.6, 0.1])
        r2 = a.tick(scores2)
        assert r2.mask[1] == 1.0
        assert r2.history_size == 0

    def test_tick_basin_stability(self) -> None:
        """Малый Δscore, stay + history_size вырос на 1."""
        a = TaskAttractor(
            n_tasks=3,
            base_dwell=1,
            dwell_slope=0.0,
            plasticity_gain=0.0,
            basin_threshold=0.2,
        )
        # Первый тик: колонка 0
        scores1 = np.array([0.5, 0.2, 0.1])
        r1 = a.tick(scores1)
        assert r1.mask[0] == 1.0
        assert r1.history_size == 0

        # Второй тик: колонка 1 выигрывает, но Δscore мал
        scores2 = np.array([0.5, 0.4, 0.1])
        # Δ = 0.4 - 0.5 = -0.1, immediate_switch: 0.1 > 0.3 → False
        # basin: -0.1 > -0.25 → True → stay
        r2 = a.tick(scores2)
        assert r2.mask[0] == 1.0
        assert r2.history_size == 1

    def test_tick_after_reset(self) -> None:
        """reset() затем tick(scores): mask = one-hot по argmax, history = 0."""
        a = TaskAttractor(n_tasks=3)
        scores = np.array([0.2, 0.7, 0.1])

        # Первый тик
        r1 = a.tick(scores)
        assert r1.mask[1] == 1.0
        assert r1.history_size == 0

        # Сброс
        a.reset()
        assert a.current_mask is None
        assert a.history_size == 0

        # После сброса — как новый объект
        r2 = a.tick(scores)
        assert r2.mask[1] == 1.0
        assert r2.history_size == 0

    def test_tick_reset_mask_none(self) -> None:
        """reset() ставит mask = None, а не zeros."""
        a = TaskAttractor(n_tasks=3)
        scores = np.array([0.2, 0.7, 0.1])
        a.tick(scores)
        a.reset()
        assert a.current_mask is None

    def test_current_mask_none_before(self) -> None:
        """current_mask = None до первого tick()."""
        a = TaskAttractor(n_tasks=3)
        assert a.current_mask is None

    def test_current_mask_after_tick(self) -> None:
        """current_mask возвращает копию mask после tick()."""
        a = TaskAttractor(n_tasks=3)
        scores = np.array([0.2, 0.7, 0.1])
        a.tick(scores)
        mask = a.current_mask
        assert mask is not None
        assert mask[1] == 1.0

    def test_reset_after_multiple_ticks(self) -> None:
        """Сброс после нескольких тиков: следующий tick работает как первый."""
        a = TaskAttractor(n_tasks=3, base_dwell=1)
        scores = np.array([0.2, 0.7, 0.1])

        for _ in range(5):
            a.tick(scores)

        # 5 тиков: первый = 0, остальные 4 инкремента → history_size = 4
        assert a.history_size == 4
        a.reset()
        assert a.history_size == 0
        assert a.current_mask is None

        r = a.tick(scores)
        assert r.history_size == 0
        assert r.mask[1] == 1.0
