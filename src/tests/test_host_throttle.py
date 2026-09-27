"""Unit tests for throttle Functional Core (src/host/throttle.py).

plan_throttle is pure: a critical homeostatic channel activates an internal,
reversible throttle plan; otherwise the plan is inactive.
"""

from __future__ import annotations

import pytest

from src.core.homeostasis import HomeostasisState, HomeostaticSignal
from src.host.throttle import plan_throttle


def _state(signals: tuple[HomeostaticSignal, ...]) -> HomeostasisState:
    """Собрать снимок гомеостаза из сигналов."""
    return HomeostasisState(
        signals=signals,
        max_deviation=max((s.deviation for s in signals), default=0.0),
        severity=max((s.severity for s in signals), default=0.0),
        is_critical=any(s.is_critical for s in signals),
    )


def _sig(
    tag: str, severity: float, deviation: float, critical: bool
) -> HomeostaticSignal:
    return HomeostaticSignal(
        tag=tag, severity=severity, deviation=deviation, is_critical=critical
    )


class TestPlanThrottle:
    """Tests for plan_throttle."""

    def test_inactive_in_norm(self) -> None:
        """Норма → throttle неактивен."""
        plan = plan_throttle(_state((_sig("battery", 0.3, 0.0, False),)))
        assert plan.active is False
        assert plan.llm_gate is False

    def test_active_on_critical(self) -> None:
        """severity >= 0.9 → throttle активен, LLM-гейт включён."""
        plan = plan_throttle(_state((_sig("battery", 0.95, 1.0, True),)))
        assert plan.active is True
        assert plan.llm_gate is True
        assert plan.k_scale == 0.5
        assert plan.dt_scale == 2.0

    def test_reason_mentions_channel(self) -> None:
        """Причина называет критический канал (аудит/explainability)."""
        plan = plan_throttle(_state((_sig("resources", 0.99, 1.0, True),)))
        assert "resources" in plan.reason

    def test_worst_channel_in_reason(self) -> None:
        """При нескольких критических — худший по severity в причине."""
        plan = plan_throttle(
            _state(
                (
                    _sig("battery", 0.91, 1.0, True),
                    _sig("resources", 0.99, 1.0, True),
                )
            )
        )
        assert "resources" in plan.reason

    def test_purity(self) -> None:
        """Одинаковый вход → одинаковый план."""
        state = _state((_sig("battery", 0.95, 1.0, True),))
        assert plan_throttle(state) == plan_throttle(state)

    def test_custom_scales(self) -> None:
        """Кастомные множители попадают в план."""
        plan = plan_throttle(
            _state((_sig("battery", 0.95, 1.0, True),)),
            k_scale=0.25,
            dt_scale=4.0,
        )
        assert plan.k_scale == 0.25
        assert plan.dt_scale == 4.0

    def test_custom_threshold(self) -> None:
        """Кастомный порог управляет активацией."""
        state = _state((_sig("battery", 0.6, 0.25, False),))
        assert plan_throttle(state, severity_threshold=0.5).active is True
        assert plan_throttle(state, severity_threshold=0.9).active is False

    def test_invalid_threshold_raises(self) -> None:
        """severity_threshold вне [0, 1] → ValueError."""
        with pytest.raises(ValueError):
            plan_throttle(_state(()), severity_threshold=1.5)

    def test_invalid_k_scale_raises(self) -> None:
        """k_scale вне (0, 1] → ValueError."""
        with pytest.raises(ValueError):
            plan_throttle(_state(()), k_scale=1.5)
        with pytest.raises(ValueError):
            plan_throttle(_state(()), k_scale=0.0)

    def test_invalid_dt_scale_raises(self) -> None:
        """dt_scale < 1 → ValueError."""
        with pytest.raises(ValueError):
            plan_throttle(_state(()), dt_scale=0.5)
