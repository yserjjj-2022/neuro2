"""Unit tests for CapabilityGate — single side-effect authorization point (S4).

Fail-safe deny: tier above max, or irreversible without HITL token → refusal.
On S4 no external irreversible action exists; the gate is a skeleton.
"""

from __future__ import annotations

from src.host.gate import (
    ActionRequest,
    Capability,
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


class TestGranularCapabilities:
    """S4-долг: гранулярные права поверх tier-лестницы."""

    def test_all_capabilities_granted_by_default(self) -> None:
        """Без явного granted разрешены все права."""
        gate = CapabilityGate()
        assert gate.granted == frozenset(Capability)

    def test_action_without_capabilities_allowed(self) -> None:
        """Действие без требуемых прав не зависит от granted (совместимость)."""
        gate = CapabilityGate(granted=frozenset())
        decision = gate.request(
            ActionRequest(
                name="speak",
                tier=CapabilityTier.T1,
                reversible=True,
                reason="respond",
            )
        )
        assert decision.allowed is True

    def test_deny_when_capability_not_granted(self) -> None:
        """Требуемое право отсутствует в granted → отказ (fail-safe deny)."""
        gate = CapabilityGate(granted=frozenset({Capability.READ}))
        decision = gate.request(
            ActionRequest(
                name="think",
                tier=CapabilityTier.T1,
                reversible=True,
                reason="initiative",
                capabilities=frozenset({Capability.THINK}),
            )
        )
        assert decision.allowed is False
        assert "missing capabilities" in decision.reason
        assert "think" in decision.reason

    def test_allow_when_capability_granted(self) -> None:
        """Требуемое право выдано → разрешено."""
        gate = CapabilityGate(granted=frozenset({Capability.SPEAK}))
        decision = gate.request(
            ActionRequest(
                name="speak",
                tier=CapabilityTier.T1,
                reversible=True,
                reason="respond",
                capabilities=frozenset({Capability.SPEAK}),
            )
        )
        assert decision.allowed is True

    def test_escape_hatch_speak_without_think(self) -> None:
        """Под throttle: SPEAK выдан, THINK отозван → голос возможен, LLM нет.

        Это и есть escape hatch: хост сообщает о перегрузке шаблоном,
        не имея права на дорогой инициативный вызов.
        """
        gate = CapabilityGate(granted=frozenset({Capability.READ, Capability.SPEAK}))
        speak = gate.request(
            ActionRequest(
                name="speak",
                tier=CapabilityTier.T1,
                reversible=True,
                reason="escape hatch",
                capabilities=frozenset({Capability.SPEAK}),
            )
        )
        think = gate.request(
            ActionRequest(
                name="initiative",
                tier=CapabilityTier.T1,
                reversible=True,
                reason="initiative",
                capabilities=frozenset({Capability.THINK}),
            )
        )
        assert speak.allowed is True
        assert think.allowed is False

    def test_revoke_and_grant(self) -> None:
        """revoke/grant меняют набор прав."""
        gate = CapabilityGate(granted=frozenset({Capability.SPEAK}))
        gate.revoke(Capability.SPEAK)
        assert Capability.SPEAK not in gate.granted
        gate.grant(Capability.SPEAK)
        assert Capability.SPEAK in gate.granted

    def test_capability_checked_before_tier(self) -> None:
        """Права проверяются раньше tier: отказ с причиной о правах."""
        gate = CapabilityGate(
            max_tier=CapabilityTier.T0, granted=frozenset({Capability.READ})
        )
        decision = gate.request(
            ActionRequest(
                name="speak",
                tier=CapabilityTier.T4,
                reversible=True,
                reason="x",
                capabilities=frozenset({Capability.SPEAK}),
            )
        )
        assert decision.allowed is False
        assert "missing capabilities" in decision.reason
