"""Unit tests for SelfMonitor (S6): observables accumulate, reset planning."""

from __future__ import annotations

import pytest

from src.core.selfcontrol import ChangeKind, Metacognition, ResetLevel, SelfMonitor


class TestSelfMonitor:
    def test_first_observe_returns_metacognition(self) -> None:
        monitor = SelfMonitor(window=10)
        meta, plan = monitor.observe(
            scores=[1.0, 0.0, 0.0], switched=False, f=1.0, partner_uncertainty=0.0
        )
        assert isinstance(meta, Metacognition)
        assert plan.triggered is False
        assert monitor.metacognition is meta

    def test_observables_accumulate(self) -> None:
        monitor = SelfMonitor(window=10)
        for _ in range(6):
            monitor.observe(
                scores=[1.0, 1.0, 1.0], switched=True, f=1.0,
                partner_uncertainty=0.5,
            )
        meta = monitor.metacognition
        assert meta is not None
        assert meta.metastability == pytest.approx(1.0)
        assert meta.epistemic_uncertainty == pytest.approx(0.5)
        assert meta.conflict == pytest.approx(1.0)

    def test_uncertainty_clamped(self) -> None:
        monitor = SelfMonitor(window=5)
        meta, _ = monitor.observe(
            scores=[1.0], switched=False, f=0.0, partner_uncertainty=5.0
        )
        assert meta.epistemic_uncertainty == 1.0

    def test_core_broken_triggers_hard(self) -> None:
        monitor = SelfMonitor(window=5)
        _, plan = monitor.observe(
            scores=[1.0], switched=False, f=1.0, partner_uncertainty=0.0,
            core_preserved=False,
        )
        assert plan.level is ResetLevel.HARD

    def test_drift_with_warning_freezes(self) -> None:
        monitor = SelfMonitor(window=5, warning_threshold=0.0)
        # warning_threshold=0.0 → slowing >= 0 всегда warning; traceable=False
        # → DRIFT → FREEZE (regime shift).
        _, plan = monitor.observe(
            scores=[1.0, 1.0], switched=True, f=1.0, partner_uncertainty=0.0,
            core_preserved=True, traceable=False, coherent=True,
        )
        assessment = monitor.last_assessment
        assert assessment is not None
        assert assessment.kind is ChangeKind.DRIFT
        assert plan.level is ResetLevel.FREEZE

    def test_stable_series_no_trigger(self) -> None:
        monitor = SelfMonitor(window=20, warning_threshold=0.99)
        for _ in range(10):
            _, plan = monitor.observe(
                scores=[1.0, 0.0], switched=False, f=1.0, partner_uncertainty=0.0
            )
        assert plan.triggered is False

    def test_bad_window_raises(self) -> None:
        with pytest.raises(ValueError):
            SelfMonitor(window=0)

    def test_bad_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            SelfMonitor(warning_threshold=2.0)
