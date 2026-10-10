"""Tests for runtime glue (ADR-0011 §6): text→vector, ProbeTransport, wiring."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from src.config import AutonomyConfig, HostConfig, MemoryConfig
from src.host.loop import build_host_loop
from src.integrations.models import (
    IntegrationKind,
    IntegrationSpec,
    LocalTransport,
    Provenance,
    StdioTransport,
)
from src.integrations.registry import IntegrationRegistry
from src.integrations.runtime import ProbeTransport, hash_text_to_vector
from src.mcp.client import MCPClient, MCPClientError
from src.mcp.models import SignalCategory
from src.mcp.probe import Affordance


class TestHashTextToVector:
    def test_deterministic(self) -> None:
        assert hash_text_to_vector("abc", 4) == hash_text_to_vector("abc", 4)

    def test_differs(self) -> None:
        assert hash_text_to_vector("abc", 4) != hash_text_to_vector("xyz", 4)

    def test_dim_and_range(self) -> None:
        vec = hash_text_to_vector("hello", 8)
        assert len(vec) == 8
        assert all(0.0 <= v <= 1.0 for v in vec)

    def test_bad_dim(self) -> None:
        with pytest.raises(ValueError):
            hash_text_to_vector("x", 0)


class _FakeClient:
    """Минимальный fake MCP-клиент (без I/O)."""

    def __init__(self, text: str = "ok", *, error: bool = False) -> None:
        self._text = text
        self._error = error
        self.last_args: dict[str, str] | None = None

    def call_tool(self, name: str, arguments: object = None) -> object:
        class _Result:
            def __init__(self, success: bool, text: str) -> None:
                self.success = success
                self.text = text

        if self._error:
            raise MCPClientError("boom")
        self.last_args = arguments  # type: ignore[assignment]
        return _Result(True, self._text)


def _routes(
    items: dict[str, tuple[_FakeClient, str, dict[str, str]]],
) -> dict[str, tuple[MCPClient, str, dict[str, str]]]:
    """Собрать маршруты ProbeTransport с fake-клиентами (приведение типа)."""
    return cast("dict[str, tuple[MCPClient, str, dict[str, str]]]", items)


def _sensor_spec() -> IntegrationSpec:
    """Минимальный локальный SENSOR — чтобы шина собралась из реестра."""
    return IntegrationSpec(
        name="battery",
        kind=IntegrationKind.SENSOR,
        transport=LocalTransport("battery"),
        category=SignalCategory.INTEROCEPTIVE,
        provenance=Provenance.LOCAL,
    )


class TestProbeTransport:
    def test_call_returns_vector(self) -> None:
        transport = ProbeTransport(_routes({"echo": (_FakeClient("hi"), "echo", {})}))
        vec = transport(Affordance("echo", SignalCategory.EXTEROCEPTIVE))
        assert len(vec) == 4
        assert vec == hash_text_to_vector("hi", 4)

    def test_arguments_forwarded(self) -> None:
        client = _FakeClient("hi")
        transport = ProbeTransport(_routes({"echo": (client, "echo", {"k": "v"})}))
        transport(Affordance("echo", SignalCategory.EXTEROCEPTIVE))
        assert client.last_args == {"k": "v"}

    def test_missing_route_empty(self) -> None:
        transport = ProbeTransport(_routes({}))
        assert transport(Affordance("nope", SignalCategory.EXTEROCEPTIVE)) == ()

    def test_error_isolated(self) -> None:
        transport = ProbeTransport(
            _routes({"echo": (_FakeClient(error=True), "echo", {})})
        )
        assert transport(Affordance("echo", SignalCategory.EXTEROCEPTIVE)) == ()

    def test_affordances_from_routes(self) -> None:
        transport = ProbeTransport(
            _routes(
                {
                    "a": (_FakeClient(), "a", {}),
                    "b": (_FakeClient(), "b", {}),
                }
            )
        )
        assert set(transport.affordances().names) == {"a", "b"}


class TestLoopWiring:
    def _config(self, tmp_path: Path) -> HostConfig:
        return HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "m.db")),
            autonomy=AutonomyConfig(enabled=True, explore_threshold=0.0),
        )

    def test_affordances_from_registry(self, tmp_path: Path) -> None:
        registry = IntegrationRegistry(
            (
                _sensor_spec(),
                IntegrationSpec(
                    name="weather",
                    kind=IntegrationKind.TOOL,
                    transport=StdioTransport("npx", ("-y", "w")),
                    provenance=Provenance.OFFICIAL,
                    tools=("get_forecast",),
                ),
            )
        )
        loop = build_host_loop(self._config(tmp_path), integrations=registry)
        assert loop.probe_effector is not None
        assert loop.probe_effector.affordances.names == ("get_forecast",)
        loop.close()

    def test_affordances_override(self, tmp_path: Path) -> None:
        from src.mcp.probe import AffordanceMap

        loop = build_host_loop(
            self._config(tmp_path),
            affordances=AffordanceMap(
                (Affordance("custom", SignalCategory.EXTEROCEPTIVE),)
            ),
        )
        assert loop.probe_effector is not None
        assert loop.probe_effector.affordances.names == ("custom",)
        loop.close()

    def test_probe_fn_used(self, tmp_path: Path) -> None:
        registry = IntegrationRegistry(
            (
                _sensor_spec(),
                IntegrationSpec(
                    name="w",
                    kind=IntegrationKind.TOOL,
                    transport=StdioTransport("npx", ("-y", "w")),
                    tools=("t",),
                ),
            )
        )
        calls: list[str] = []

        def fake_probe(affordance: Affordance) -> tuple[float, ...]:
            calls.append(affordance.name)
            return (1.0,)

        loop = build_host_loop(
            self._config(tmp_path), integrations=registry, probe_fn=fake_probe
        )
        loop.step_once(0)
        result = loop.explore()
        assert result is not None and result.success
        assert calls == ["t"]
        loop.close()

    def test_s6_compat_no_registry(self, tmp_path: Path) -> None:
        loop = build_host_loop(self._config(tmp_path))
        assert loop.probe_effector is not None
        # Без реестра — прежний default_affordances().
        assert "web_search" in loop.probe_effector.affordances.names
        loop.close()
