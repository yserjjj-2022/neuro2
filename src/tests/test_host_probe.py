"""Tests for gated MCP probe effector and loop.explore wiring (S6 проход 2)."""

from __future__ import annotations

from pathlib import Path

from src.config import AutonomyConfig, HostConfig, MemoryConfig
from src.host.gate import Capability, CapabilityGate, CapabilityTier
from src.host.loop import HostLoop, build_host_loop
from src.host.probe import ProbeEffector
from src.mcp import (
    Affordance,
    AffordanceMap,
    ProbeRequest,
    SignalCategory,
    default_affordances,
)


def _effector(
    *,
    granted: frozenset[Capability] | None = None,
    max_tier: CapabilityTier = CapabilityTier.T4,
    affordances: AffordanceMap | None = None,
) -> ProbeEffector:
    return ProbeEffector(
        affordances=affordances if affordances is not None else default_affordances(),
        gate=CapabilityGate(max_tier=max_tier, granted=granted),
    )


class TestProbeEffector:
    def test_reversible_probe_allowed(self) -> None:
        effector = _effector(
            granted=frozenset({Capability.READ, Capability.ACT_REVERSIBLE})
        )
        result = effector.probe(ProbeRequest("web_search", "test"))
        assert result.success
        assert result.affordance == "web_search"
        assert len(result.data) == 4

    def test_unknown_affordance_denied(self) -> None:
        effector = _effector()
        result = effector.probe(ProbeRequest("nope", "test"))
        assert not result.success
        assert "unknown" in result.reason

    def test_irreversible_without_hitl_denied(self) -> None:
        amap = AffordanceMap(
            (Affordance("act", SignalCategory.EXTEROCEPTIVE, reversible=False),)
        )
        effector = _effector(
            granted=frozenset(
                {Capability.READ, Capability.ACT_IRREVERSIBLE}
            ),
            affordances=amap,
        )
        denied = effector.probe(ProbeRequest("act", "test"))
        assert not denied.success
        allowed = effector.probe(ProbeRequest("act", "test"), hitl_token="ok")
        assert allowed.success

    def test_missing_capability_denied(self) -> None:
        effector = _effector(granted=frozenset({Capability.READ}))
        result = effector.probe(ProbeRequest("web_search", "test"))
        assert not result.success
        assert "missing capabilities" in result.reason

    def test_transport_error_is_safe(self) -> None:
        def boom(_affordance: Affordance) -> tuple[float, ...]:
            raise RuntimeError("network down")

        effector = ProbeEffector(
            affordances=default_affordances(),
            gate=CapabilityGate(
                max_tier=CapabilityTier.T4,
                granted=frozenset({Capability.READ, Capability.ACT_REVERSIBLE}),
            ),
            probe_fn=boom,
        )
        result = effector.probe(ProbeRequest("web_search", "test"))
        assert not result.success
        assert "probe error" in result.reason


class TestLoopExplore:
    def _loop(self, tmp_path: Path) -> HostLoop:
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "m.db")),
            autonomy=AutonomyConfig(enabled=True, explore_threshold=0.0),
        )
        return build_host_loop(config)

    def test_explore_requires_effector(self, tmp_path: Path) -> None:
        loop = self._loop(tmp_path)
        assert loop.probe_effector is not None
        loop.step_once(0)
        result = loop.explore()
        assert result is not None
        loop.close()

    def test_explore_disabled_without_autonomy(self, tmp_path: Path) -> None:
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "m.db")),
            autonomy=AutonomyConfig(enabled=False),
        )
        loop = build_host_loop(config)
        assert loop.probe_effector is None
        assert loop.explore() is None
        loop.close()

    def test_probe_reaches_telemetry(self, tmp_path: Path) -> None:
        loop = self._loop(tmp_path)
        loop.step_once(0)
        loop.explore()
        loop.step_once(1)
        loop.close()
        text = (tmp_path / "run.jsonl").read_text(encoding="utf-8")
        assert '"probe_affordance": "web_search"' in text
        assert '"probe_success": true' in text
