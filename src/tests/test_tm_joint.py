"""Unit tests for JointAgency — shared goals, backstop not obey (S5, проход 2)."""

from __future__ import annotations

import pytest

from src.tm import JointAgency, JointGoal


class TestJointGoal:
    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            JointGoal(task="x", ttl_ticks=0)
        with pytest.raises(ValueError):
            JointGoal(task="x", priority=1.5)


class TestJointAgency:
    def test_propose_creates_goal(self) -> None:
        agency = JointAgency(default_ttl=10)
        goal = agency.propose("build", tick=0, priority=0.8)
        assert goal.task == "build"
        assert agency.goal is goal

    def test_expires_by_ttl(self) -> None:
        agency = JointAgency(default_ttl=5)
        agency.propose("build", tick=0)
        assert agency.expired(4) is False
        assert agency.expired(5) is True

    def test_drift_returns_reminder(self) -> None:
        """Уход от общей цели → напоминание, а не приказ."""
        agency = JointAgency(default_ttl=10)
        agency.propose("build", tick=0)
        reminder = agency.update("chat", tick=1)
        assert reminder is not None
        assert "build" in reminder

    def test_on_track_no_reminder(self) -> None:
        agency = JointAgency(default_ttl=10)
        agency.propose("build", tick=0)
        assert agency.update("build", tick=1) is None

    def test_expired_clears_goal(self) -> None:
        agency = JointAgency(default_ttl=3)
        agency.propose("build", tick=0)
        assert agency.update("chat", tick=5) is None
        assert agency.goal is None

    def test_no_goal_no_reminder(self) -> None:
        agency = JointAgency()
        assert agency.update("chat", tick=0) is None
