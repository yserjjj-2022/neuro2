"""Unit tests for energy guards and drift detector."""

from __future__ import annotations

import pytest

from src.core.energy.drift import DriftDetector
from src.core.energy.guards import HostIntegrityError, check_finite
from src.core.energy.models import FreeEnergyResult


def _result(
    f: float = 1.0,
    valence: float = -1.0,
    allostatic_stress: float = 1.0,
    gamma: float = 1.0,
) -> FreeEnergyResult:
    return FreeEnergyResult(
        f=f, valence=valence, allostatic_stress=allostatic_stress, gamma=gamma
    )


class TestCheckFinite:
    """Guard конечности."""

    def test_valid_passes(self) -> None:
        check_finite(_result())

    @pytest.mark.parametrize("field", ["f", "valence", "allostatic_stress", "gamma"])
    def test_nan_raises(self, field: str) -> None:
        kwargs = {field: float("nan")}
        with pytest.raises(HostIntegrityError):
            check_finite(_result(**kwargs))

    @pytest.mark.parametrize("field", ["f", "valence", "allostatic_stress", "gamma"])
    def test_inf_raises(self, field: str) -> None:
        kwargs = {field: float("inf")}
        with pytest.raises(HostIntegrityError):
            check_finite(_result(**kwargs))

    def test_message_names_field(self) -> None:
        with pytest.raises(HostIntegrityError, match="valence"):
            check_finite(_result(valence=float("nan")))


class TestDriftDetector:
    """Детектор дрейфа."""

    def test_no_drift_when_below_threshold(self) -> None:
        det = DriftDetector(f_threshold=10.0, stress_threshold=10.0, hold_ticks=3)
        for _ in range(10):
            assert det.update(_result(f=1.0, allostatic_stress=1.0)) is False
        assert det.streak == 0

    def test_drift_after_hold(self) -> None:
        det = DriftDetector(f_threshold=1.0, stress_threshold=1.0, hold_ticks=3)
        high = _result(f=5.0, allostatic_stress=5.0)
        assert det.update(high) is False  # streak=1
        assert det.update(high) is False  # streak=2
        assert det.update(high) is False  # streak=3
        assert det.update(high) is True  # streak=4 > 3

    def test_reset_on_recovery(self) -> None:
        det = DriftDetector(f_threshold=1.0, stress_threshold=1.0, hold_ticks=2)
        det.update(_result(f=5.0, allostatic_stress=5.0))
        det.update(_result(f=5.0, allostatic_stress=5.0))
        det.update(_result(f=0.0, allostatic_stress=0.0))  # recovery
        assert det.streak == 0
        assert det.update(_result(f=5.0, allostatic_stress=5.0)) is False

    def test_stress_alone_triggers(self) -> None:
        det = DriftDetector(f_threshold=100.0, stress_threshold=1.0, hold_ticks=1)
        det.update(_result(f=0.0, allostatic_stress=5.0))
        assert det.update(_result(f=0.0, allostatic_stress=5.0)) is True

    def test_invalid_hold_raises(self) -> None:
        with pytest.raises(ValueError):
            DriftDetector(f_threshold=1.0, stress_threshold=1.0, hold_ticks=0)

    def test_reset_method(self) -> None:
        det = DriftDetector(f_threshold=1.0, stress_threshold=1.0, hold_ticks=1)
        det.update(_result(f=5.0, allostatic_stress=5.0))
        det.reset()
        assert det.streak == 0
