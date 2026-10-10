"""Tests for catalog → runtime bridges (ADR-0011 §6): provider + affordances."""

from __future__ import annotations

import pytest

from src.host.sources import (
    BatteryProvider,
    CircadianProvider,
    CpuProvider,
    SignalProvider,
    UserMessageProvider,
)
from src.integrations.bridges import to_affordances
from src.integrations.factories import (
    SensorContext,
    build_provider,
    build_providers,
    enabled_sensors,
)
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


def _sensor(provider: str, **overrides: object) -> IntegrationSpec:
    base: dict[str, object] = {
        "name": provider,
        "kind": IntegrationKind.SENSOR,
        "transport": LocalTransport(provider),
        "provenance": Provenance.LOCAL,
    }
    base.update(overrides)
    return IntegrationSpec(**base)  # type: ignore[arg-type]


class TestBuildProvider:
    def test_local_circadian(self) -> None:
        provider = build_provider(_sensor("circadian"), SensorContext())
        assert isinstance(provider, CircadianProvider)
        assert isinstance(provider, SignalProvider)

    def test_local_battery(self) -> None:
        provider = build_provider(
            _sensor("battery", category=SignalCategory.INTEROCEPTIVE),
            SensorContext(),
        )
        assert isinstance(provider, BatteryProvider)
        assert provider.category is SignalCategory.INTEROCEPTIVE

    def test_local_cpu_uses_seed(self) -> None:
        provider = build_provider(_sensor("cpu"), SensorContext(seed=7))
        assert isinstance(provider, CpuProvider)
        assert provider.seed == 7

    def test_message_falls_back_to_stub(self) -> None:
        """Без инъекции — заглушка UserMessageProvider (message вне памяти)."""
        provider = build_provider(
            _sensor("message"), SensorContext(message_dim=4, seed=1)
        )
        assert isinstance(provider, UserMessageProvider)
        assert provider.dim == 4

    def test_message_uses_injected_provider(self) -> None:
        injected = UserMessageProvider(embedding_dim=16, seed=2)
        provider = build_provider(
            _sensor("message"), SensorContext(message_provider=injected)
        )
        assert provider is injected

    def test_resources_requires_injection(self) -> None:
        with pytest.raises(NotImplementedError):
            build_provider(_sensor("resources"), SensorContext())

    def test_non_sensor_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_provider(_spec("x"), SensorContext())

    def test_unknown_local_provider(self) -> None:
        with pytest.raises(NotImplementedError):
            build_provider(_sensor("weird"), SensorContext())

    def test_mcp_sensor_not_implemented(self) -> None:
        spec = _spec(
            "remote",
            kind=IntegrationKind.SENSOR,
            transport=StdioTransport("npx", ("-y", "x")),
        )
        with pytest.raises(NotImplementedError):
            build_provider(spec, SensorContext())


class TestBuildProviders:
    def test_order_preserved(self) -> None:
        """Порядок провайдеров = порядок SENSOR-записей (порядок укладки)."""
        specs = (
            _sensor("circadian"),
            _sensor("battery"),
            _sensor("cpu"),
            _sensor("message"),
        )
        providers = build_providers(specs, SensorContext())
        assert [p.tag for p in providers] == [
            "circadian",
            "battery",
            "cpu",
            "user_message",
        ]

    def test_non_sensors_and_disabled_skipped(self) -> None:
        specs = (
            _spec("tool"),  # TOOL — не сенсор
            _sensor("battery"),
            _sensor("cpu", enabled=False),
        )
        providers = build_providers(specs, SensorContext())
        assert [p.tag for p in providers] == ["battery"]

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            build_providers((_spec("tool"),), SensorContext())

    def test_enabled_sensors_filter(self) -> None:
        specs = (_sensor("battery"), _sensor("cpu", enabled=False), _spec("t"))
        assert [s.name for s in enabled_sensors(specs)] == ["battery"]


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
        found = amap.find("w")
        assert found is not None
        assert found.category is SignalCategory.EXTEROCEPTIVE

    def test_duplicate_fail_fast(self) -> None:
        with pytest.raises(ValueError):
            to_affordances((_spec("a", tools=("same",)), _spec("b", tools=("same",))))
