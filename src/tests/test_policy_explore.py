"""Unit tests for policy EXPLORE / metacognition (S6)."""

from __future__ import annotations

import pytest

from src.core.homeostasis import HomeostasisState
from src.core.policy import (
    Action,
    PolicyContext,
    Preferences,
    evaluate_candidates,
    select_action,
)
from src.core.selfcontrol import Metacognition


def _homeostasis() -> HomeostasisState:
    return HomeostasisState(
        signals=(), max_deviation=0.0, severity=0.0, is_critical=False
    )


def _context(
    *,
    metacognition: Metacognition | None,
    has_new_message: bool = False,
    f: float = 0.0,
) -> PolicyContext:
    return PolicyContext(
        f=f,
        valence=0.0,
        stress=0.0,
        task="tone",
        homeostasis=_homeostasis(),
        has_new_message=has_new_message,
        metacognition=metacognition,
    )


class TestEvaluateExplore:
    def test_high_uncertainty_wins(self) -> None:
        ctx = _context(metacognition=Metacognition(0.2, 0.2, 0.9, 0.1))
        prefs = Preferences(pragmatic_weight=1.0, epistemic_weight=1.0)
        trace = select_action(ctx, prefs)
        assert trace.chosen is Action.EXPLORE

    def test_low_uncertainty_no_explore(self) -> None:
        ctx = _context(metacognition=Metacognition(0.1, 0.1, 0.1, 0.1))
        prefs = Preferences(explore_threshold=0.6)
        candidates = {c.action: c for c in evaluate_candidates(ctx, prefs)}
        assert candidates[Action.EXPLORE].epistemic == 0.0

    def test_message_defers_explore(self) -> None:
        ctx = _context(
            metacognition=Metacognition(0.9, 0.9, 0.9, 0.9), has_new_message=True
        )
        prefs = Preferences()
        candidates = {c.action: c for c in evaluate_candidates(ctx, prefs)}
        assert candidates[Action.EXPLORE].value == 0.0

    def test_none_metacognition_s5_compat(self) -> None:
        ctx = _context(metacognition=None)
        prefs = Preferences()
        candidates = {c.action: c for c in evaluate_candidates(ctx, prefs)}
        assert candidates[Action.EXPLORE].value == 0.0
        assert candidates[Action.EXPLORE].reason == "no metacognition (S5)"

    def test_deterministic(self) -> None:
        ctx = _context(metacognition=Metacognition(0.5, 0.5, 0.8, 0.5))
        prefs = Preferences()
        assert select_action(ctx, prefs) == select_action(ctx, prefs)

    def test_explore_threshold_validation(self) -> None:
        with pytest.raises(ValueError):
            Preferences(explore_threshold=1.5)
