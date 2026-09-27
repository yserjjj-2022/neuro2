"""Unit tests for homeostasis Functional Core (compute.py).

setpoint_deviation is pure: identical inputs → identical outputs, clip [0, 1].
"""

from __future__ import annotations

import pytest

from src.core.homeostasis.compute import setpoint_deviation


class TestSetpointDeviation:
    """Tests for setpoint_deviation — normalized deviation from a setpoint."""

    def test_below_comfort_is_zero(self) -> None:
        """severity <= comfort → 0.0 (канал в норме)."""
        assert setpoint_deviation(0.3, 0.5, 0.9) == 0.0

    def test_at_comfort_is_zero(self) -> None:
        """severity == comfort → 0.0."""
        assert setpoint_deviation(0.5, 0.5, 0.9) == 0.0

    def test_at_critical_is_one(self) -> None:
        """severity == critical → 1.0."""
        assert setpoint_deviation(0.9, 0.5, 0.9) == 1.0

    def test_above_critical_clipped(self) -> None:
        """severity > critical → 1.0 (клип)."""
        assert setpoint_deviation(1.0, 0.5, 0.9) == 1.0

    def test_midpoint(self) -> None:
        """Середина диапазона → 0.5."""
        assert setpoint_deviation(0.7, 0.5, 0.9) == pytest.approx(0.5)

    def test_linear_between(self) -> None:
        """Линейность между comfort и critical."""
        assert setpoint_deviation(0.6, 0.5, 0.9) == pytest.approx(0.25)
        assert setpoint_deviation(0.8, 0.5, 0.9) == pytest.approx(0.75)

    def test_purity(self) -> None:
        """Одинаковый вход → одинаковый выход (детерминизм)."""
        assert setpoint_deviation(0.7, 0.5, 0.9) == setpoint_deviation(0.7, 0.5, 0.9)

    def test_critical_leq_comfort_raises(self) -> None:
        """critical <= comfort → ValueError."""
        with pytest.raises(ValueError):
            setpoint_deviation(0.5, 0.9, 0.9)
        with pytest.raises(ValueError):
            setpoint_deviation(0.5, 0.9, 0.5)

    def test_zero_width_range_raises(self) -> None:
        """critical == comfort → ValueError (деление на ноль недопустимо)."""
        with pytest.raises(ValueError):
            setpoint_deviation(0.5, 0.5, 0.5)
