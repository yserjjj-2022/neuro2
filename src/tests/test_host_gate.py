"""Unit tests for CapabilityGate — single side-effect authorization point (S4).

Fail-safe deny: tier above max, or irreversible without HITL token → refusal.
On S4 no external irreversible action exists; the gate is a skeleton.
"""

from __future__ import annotations

from src.host.gate import (
    ActionRequest,
    CapabilityGate,
    CapabilityTier,
    GateDecision,
)


class TestCapabilityGate:
    """Tests for CapabilityGate.request."""

    def test_allow_speech_t1(self) -> None:
        """Речь (T1) при max_tier=T1 → разрешено."""
        gate = CapabilityGate(max_tier=CapabilityTier.T1)
        decision = gate.request(
            ActionRequest(
                name="speak",
                tier=CapabilityTier.T1,
                reversible=True,
                reason="respond",
            )
        )
        assert decision.allowed is True
        assert "speak" in decision.reason

    def test_allow_observation_t0(self) -> None:
        """Наблюдение (T0) → разрешено (ниже max)."""
        gate = CapabilityGate(max_tier=CapabilityTier.T1)
        decision = gate.request(
            ActionRequest(
                name="observe",
                tier=CapabilityTier.T0,
                reversible=True,
                reason="sense",
            )
        )
        assert decision.allowed is True

    def test_deny_above_max_tier(self) -> None:
        """tier > max_tier → отказ (fail-safe deny)."""
        gate = CapabilityGate(max_tier=CapabilityTier.T1)
        decision = gate.request(
            ActionRequest(
                name="mcp_write",
                tier=CapabilityTier.T2,
                reversible=True,
                reason="tool call",
            )
        )
        assert decision.allowed is False
        assert "tier" in decision.reason

    def test_deny_irreversible_without_hitl(self) -> None:
        """Необратимое без HITL-токена → отказ (fail-safe deny)."""
        gate = CapabilityGate(max_tier=CapabilityTier.T4)
        decision = gate.request(
            ActionRequest(
                name="delete",
                tier=CapabilityTier.T3,
                reversible=False,
                reason="cleanup",
            )
        )
        assert decision.allowed is False
        assert "HITL" in decision.reason

    def test_allow_irreversible_with_hitl(self) -> None:
        """Необратимое с HITL-токеном (в пределах tier) → разрешено."""
        gate = CapabilityGate(max_tier=CapabilityTier.T4)
        decision = gate.request(
            ActionRequest(
                name="delete",
                tier=CapabilityTier.T4,
                reversible=False,
                reason="cleanup",
                hitl_token="approved",
            )
        )
        assert decision.allowed is True

    def test_reversible_needs_no_hitl(self) -> None:
        """Обратимое не требует HITL."""
        gate = CapabilityGate(max_tier=CapabilityTier.T3)
        decision = gate.request(
            ActionRequest(
                name="throttle",
                tier=CapabilityTier.T3,
                reversible=True,
                reason="overload",
            )
        )
        assert decision.allowed is True

    def test_default_max_tier_is_speech(self) -> None:
        """Дефолтный gate разрешает только T1 (речь)."""
        gate = CapabilityGate()
        assert gate.max_tier is CapabilityTier.T1

    def test_decision_is_frozen(self) -> None:
        """GateDecision — frozen dataclass (immutable снимок решения)."""
        decision = GateDecision(allowed=True, reason="x")
        try:
            decision.allowed = False  # type: ignore[misc]
        except AttributeError:
            pass
        else:  # pragma: no cover
            raise AssertionError("GateDecision should be frozen")
