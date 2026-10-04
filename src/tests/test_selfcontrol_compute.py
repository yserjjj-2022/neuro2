"""Unit tests for selfcontrol Core (S6): observables, CSD, classify, reset."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.selfcontrol import (
    ChangeAssessment,
    ChangeKind,
    CriticalSlowingDown,
    ResetLevel,
    classify_change,
    compute_conflict,
    compute_metastability,
    compute_saturation,
    critical_slowing_down,
    plan_reset,
)


class TestComputeConflict:
    def test_single_dominant_low(self) -> None:
        assert compute_conflict([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_uniform_high(self) -> None:
        conflict = compute_conflict([1.0, 1.0, 1.0, 1.0])
        assert conflict == pytest.approx(1.0)

    def test_two_peaks_higher_than_one(self) -> None:
        one = compute_conflict([1.0, 0.01, 0.01, 0.01])
        two = compute_conflict([1.0, 1.0, 0.01, 0.01])
        assert two > one

    def test_empty_and_single(self) -> None:
        assert compute_conflict([]) == 0.0
        assert compute_conflict([5.0]) == 0.0

    def test_all_zero(self) -> None:
        assert compute_conflict([0.0, 0.0, 0.0]) == 0.0

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_conflict([1.0, -1.0])


class TestComputeMetastability:
    def test_no_switches(self) -> None:
        assert compute_metastability([False] * 10, window=10) == 0.0

    def test_all_switches(self) -> None:
        assert compute_metastability([True] * 10, window=10) == pytest.approx(1.0)

    def test_half(self) -> None:
        flags = [True, False] * 5
        assert compute_metastability(flags, window=10) == pytest.approx(0.5)

    def test_window_limits(self) -> None:
        flags = [True] + [False] * 20
        assert compute_metastability(flags, window=5) == 0.0

    def test_empty(self) -> None:
        assert compute_metastability([], window=5) == 0.0

    def test_bad_window(self) -> None:
        with pytest.raises(ValueError):
            compute_metastability([True], window=0)


class TestComputeSaturation:
    def test_empty(self) -> None:
        assert compute_saturation([]) == 0.0

    def test_constant_series(self) -> None:
        # порог mean+std == значение → доля выше = 0.0
        assert compute_saturation([5.0] * 10) == 0.0

    def test_high_tail(self) -> None:
        series = [0.0] * 9 + [100.0]
        assert compute_saturation(series) == pytest.approx(0.1)

    def test_explicit_threshold(self) -> None:
        assert compute_saturation([1.0, 2.0, 3.0], threshold=2.5) == pytest.approx(
            1 / 3
        )


class TestCriticalSlowingDown:
    def test_short_series_safe_default(self) -> None:
        csd = critical_slowing_down([1.0, 2.0])
        assert csd.slowing == 0.0
        assert csd.is_warning is False

    def test_constant_series_low(self) -> None:
        csd = critical_slowing_down([3.0] * 20)
        assert csd.slowing < 0.5

    def test_growing_variance_raises_slowing(self) -> None:
        rng = np.random.default_rng(0)
        calm = list(1.0 + rng.normal(0, 0.01, 40))
        wild = list(1.0 + np.cumsum(rng.normal(0, 0.5, 40)))
        assert critical_slowing_down(wild).slowing > critical_slowing_down(calm).slowing

    def test_negative_gain_raises(self) -> None:
        with pytest.raises(ValueError):
            critical_slowing_down([1.0, 2.0, 3.0], variance_gain=-1.0)

    def test_deterministic(self) -> None:
        series = [1.0, 2.0, 1.5, 2.5, 3.0]
        a = critical_slowing_down(series)
        b = critical_slowing_down(series)
        assert a == b


class TestClassifyChange:
    def test_stable_when_unchanged(self) -> None:
        result = classify_change(core_preserved=True, traceable=True, coherent=True,
                                 changed=False)
        assert result.kind is ChangeKind.STABLE

    def test_all_axes_development(self) -> None:
        result = classify_change(core_preserved=True, traceable=True, coherent=True)
        assert result.kind is ChangeKind.DEVELOPMENT

    def test_any_axis_broken_drift(self) -> None:
        for kwargs in (
            {"core_preserved": False, "traceable": True, "coherent": True},
            {"core_preserved": True, "traceable": False, "coherent": True},
            {"core_preserved": True, "traceable": True, "coherent": False},
        ):
            assert classify_change(**kwargs).kind is ChangeKind.DRIFT

    def test_reason_mentions_violation(self) -> None:
        result = classify_change(core_preserved=False, traceable=True, coherent=True)
        assert "core" in result.reason


class TestPlanReset:
    def _slowing(self, *, warning: bool, value: float = 0.8) -> CriticalSlowingDown:
        return CriticalSlowingDown(
            variance=1.0, autocorrelation=0.5, slowing=value, is_warning=warning
        )

    def _assess(self, kind: ChangeKind) -> ChangeAssessment:
        return ChangeAssessment(kind, True, True, True, "test")

    def test_no_warning_no_trigger(self) -> None:
        plan = plan_reset(
            slowing=self._slowing(warning=False, value=0.1),
            assessment=self._assess(ChangeKind.STABLE),
        )
        assert plan.triggered is False

    def test_warning_soft(self) -> None:
        plan = plan_reset(
            slowing=self._slowing(warning=True),
            assessment=self._assess(ChangeKind.DEVELOPMENT),
        )
        assert plan.triggered is True
        assert plan.level is ResetLevel.SOFT

    def test_drift_warning_freeze(self) -> None:
        plan = plan_reset(
            slowing=self._slowing(warning=True),
            assessment=self._assess(ChangeKind.DRIFT),
        )
        assert plan.level is ResetLevel.FREEZE

    def test_core_broken_hard(self) -> None:
        plan = plan_reset(
            slowing=self._slowing(warning=False),
            assessment=self._assess(ChangeKind.DRIFT),
            hard_core_broken=True,
        )
        assert plan.level is ResetLevel.HARD
        assert plan.triggered is True

    def test_bad_threshold(self) -> None:
        with pytest.raises(ValueError):
            plan_reset(
                slowing=self._slowing(warning=False),
                assessment=self._assess(ChangeKind.STABLE),
                soft_threshold=1.5,
            )
