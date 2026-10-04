"""Tests for catalog → runtime bridges (ADR-0011 §6): provider + affordances."""

from __future__ import annotations

import pytest

from src.host.sources import CircadianProvider, SignalProvider
from src.integrations.bridges import to_affordances, to_provider
from src.integrations.models import (
    IntegrationKind,
    IntegrationSpec,
    LocalTransport,
    Provenance,
    StdioTransport,
)
from src.mcp.models import SignalCategory


def _spec(name: str, **overrides: object) -> IntegrationSpec:
    base: dict[str, object] = {
        "name": name,
        "kind": IntegrationKind.TOOL,
        "transport": StdioTransport("npx", ("-y", name)),
    }
    base.update(overrides)
    return IntegrationSpec(**base)  # type: ignore[arg-type]


class TestToProvider:
    def test_local_circadian(self) -> None:
        spec = _spec(
            "circadian",
            kind=IntegrationKind.SENSOR,
            transport=LocalTransport("circadian"),
            provenance=Provenance.LOCAL,
        )
        provider = to_provider(spec)
        assert isinstance(provider, CircadianProvider)
        assert isinstance(provider, SignalProvider)

    def test_non_sensor_rejected(self) -> None:
        with pytest.raises(ValueError):
            to_provider(_spec("x"))

    def test_unknown_local_provider(self) -> None:
        spec = _spec(
            "weird",
            kind=IntegrationKind.SENSOR,
            transport=LocalTransport("weird"),
        )
        with pytest.raises(NotImplementedError):
            to_provider(spec)

    def test_mcp_sensor_not_implemented(self) -> None:
        spec = _spec(
            "remote",
            kind=IntegrationKind.SENSOR,
            transport=StdioTransport("npx", ("-y", "x")),
        )
        with pytest.raises(NotImplementedError):
            to_provider(spec)


class TestToAffordances:
    def test_single_name(self) -> None:
        amap = to_affordances((_spec("weather"),))
        assert amap.names == ("weather",)

    def test_explicit_tools(self) -> None:
        amap = to_affordances((_spec("weather", tools=("get_forecast", "get_alerts")),))
        assert amap.names == ("get_forecast", "get_alerts")

    def test_disabled_skipped(self) -> None:
        amap = to_affordances((_spec("off", enabled=False), _spec("on")))
        assert amap.names == ("on",)

    def test_non_tool_skipped(self) -> None:
        amap = to_affordances(
            (
                _spec(
                    "circadian",
                    kind=IntegrationKind.SENSOR,
                    transport=LocalTransport("circadian"),
                ),
                _spec("weather"),
            )
        )
        assert amap.names == ("weather",)

    def test_category_preserved(self) -> None:
        amap = to_affordances(
            (_spec("w", category=SignalCategory.EXTEROCEPTIVE),)
        )
        assert amap.find("w") is not None
        assert amap.find("w").category is SignalCategory.EXTEROCEPTIVE

    def test_duplicate_fail_fast(self) -> None:
        with pytest.raises(ValueError):
            to_affordances((_spec("a", tools=("same",)), _spec("b", tools=("same",))))
