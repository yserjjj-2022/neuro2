"""Unit tests for attractors Functional Core (compute.py).

Tests compute_dwell, check_basin_stability, check_immediate_switch —
all pure functions with identical inputs → identical outputs.
"""

from src.core.attractors.compute import (
    check_basin_stability,
    check_immediate_switch,
    compute_dwell,
)


class TestComputeDwell:
    """Tests for compute_dwell — hysteresis + STP dwell time."""

    def test_purity(self) -> None:
        """Входные параметры не мутируются."""
        base = 5
        slope = 2.0
        gain = 0.1
        current = 0.5
        runner = 0.3
        history = 3
        result = compute_dwell(base, slope, gain, current, runner, history)
        assert result == compute_dwell(base, slope, gain, current, runner, history)

    def test_win(self) -> None:
        """Δscore > 0 → min_dwell > base_dwell."""
        assert compute_dwell(5, 2.0, 0.0, 0.5, 0.3, 0) == 5

    def test_lose(self) -> None:
        """Δscore < 0 → min_dwell == base_dwell (пол держит)."""
        assert compute_dwell(5, 2.0, 0.0, 0.5, 0.8, 0) == 5

    def test_equal(self) -> None:
        """Δscore == 0 → min_dwell == base_dwell."""
        assert compute_dwell(5, 2.0, 0.0, 0.5, 0.5, 0) == 5

    def test_stp(self) -> None:
        """history_size > 0, gain > 0 → min_dwell > base_dwell."""
        assert compute_dwell(5, 0.0, 0.1, 0.5, 0.5, 5) == 6

    def test_floor(self) -> None:
        """Δscore << 0 → min_dwell >= base_dwell (жёсткий пол)."""
        assert compute_dwell(5, 2.0, 0.0, 0.5, 2.0, 0) == 5

    def test_floor_extreme(self) -> None:
        """Экстремальный проигрыш: dwell не падает ниже base."""
        assert compute_dwell(5, 10.0, 0.0, 0.5, 10.0, 0) == 5

    def test_stp_with_win(self) -> None:
        """STP + выигрыш: оба механизма складываются."""
        assert compute_dwell(5, 2.0, 0.1, 0.5, 0.3, 5) == 6

    def test_negative_base_dwell_raises(self) -> None:
        """base_dwell < 0 → странный результат (валидация в __init__)."""
        assert compute_dwell(-1, 2.0, 0.0, 0.5, 0.8, 0) == -1


class TestCheckBasinStability:
    """Tests for check_basin_stability — basin of attraction check."""

    def test_in_basin_small_delta(self) -> None:
        """Малый Δscore → True (в бассейне)."""
        assert check_basin_stability(0.15, 0.5, 0.4, 0) is True

    def test_out_of_basin_large_delta(self) -> None:
        """Большой Δscore → False (вне бассейна)."""
        assert check_basin_stability(0.15, 0.5, 0.8, 0) is False

    def test_stp_boost(self) -> None:
        """history_size растит effective_threshold."""
        assert check_basin_stability(0.15, 0.5, 0.8, 0) is False
        assert check_basin_stability(0.15, 0.5, 0.8, 5) is True

    def test_equal_scores(self) -> None:
        """Δscore == 0 → всегда в бассейне."""
        assert check_basin_stability(0.15, 0.5, 0.5, 0) is True

    def test_large_basin_threshold(self) -> None:
        """Большой порог → больше стабильности."""
        assert check_basin_stability(1.0, 0.5, 0.8, 0) is True


class TestCheckImmediateSwitch:
    """Tests for check_immediate_switch — competitor dominance check."""

    def test_dominance_true(self) -> None:
        """Большой Δscore (конкурент явный) → True."""
        assert check_immediate_switch(0.5, 0.81, 0.3) is True

    def test_dominance_strong(self) -> None:
        """Сильное превосходство → True."""
        assert check_immediate_switch(0.5, 0.9, 0.3) is True

    def test_no_dominance_small(self) -> None:
        """Малый Δscore → False."""
        assert check_immediate_switch(0.5, 0.55, 0.3) is False

    def test_dominance_equal(self) -> None:
        """Равные scores → False."""
        assert check_immediate_switch(0.5, 0.5, 0.3) is False

    def test_default_threshold(self) -> None:
        """Дефолтный порог 0.3 работает."""
        assert check_immediate_switch(0.5, 0.9) is True

    def test_custom_threshold(self) -> None:
        """Кастомный порог."""
        assert check_immediate_switch(0.5, 0.7, 0.15) is True
