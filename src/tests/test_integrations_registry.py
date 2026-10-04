"""Unit tests for IntegrationRegistry and default roster (ADR-0011)."""

from __future__ import annotations

import pytest

from src.integrations.models import (
    IntegrationKind,
    IntegrationSpec,
    Provenance,
    StdioTransport,
)
from src.integrations.registry import (
    IntegrationRegistry,
    default_integrations,
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


class TestIntegrationRegistry:
    def test_empty(self) -> None:
        assert len(IntegrationRegistry()) == 0

    def test_add_and_find(self) -> None:
        reg = IntegrationRegistry()
        reg.add(_spec("a"))
        assert reg.find("a") is not None
        assert reg.find("missing") is None
        assert len(reg) == 1

    def test_duplicate_rejected(self) -> None:
        reg = IntegrationRegistry((_spec("a"),))
        with pytest.raises(ValueError):
            reg.add(_spec("a"))

    def test_by_kind(self) -> None:
        reg = IntegrationRegistry(
            (
                _spec("t", kind=IntegrationKind.TOOL),
                _spec("s", kind=IntegrationKind.SENSOR),
            )
        )
        assert [s.name for s in reg.by_kind(IntegrationKind.SENSOR)] == ["s"]
        assert [s.name for s in reg.by_kind(IntegrationKind.TOOL)] == ["t"]

    def test_by_category(self) -> None:
        reg = IntegrationRegistry(
            (
                _spec("e", category=SignalCategory.EXTEROCEPTIVE),
                _spec("i", category=SignalCategory.INTEROCEPTIVE),
            )
        )
        assert [s.name for s in reg.by_category(SignalCategory.INTEROCEPTIVE)] == [
            "i"
        ]

    def test_enabled_filter(self) -> None:
        reg = IntegrationRegistry((_spec("on"), _spec("off", enabled=False)))
        assert [s.name for s in reg.enabled()] == ["on"]


class TestDefaultIntegrations:
    def test_roster_names(self) -> None:
        names = [s.name for s in default_integrations()]
        assert names == [
            "circadian",
            "resources",
            "everything",
            "time",
            "weather",
            "web-search",
        ]

    def test_official_first_community_disabled(self) -> None:
        reg = IntegrationRegistry(default_integrations())
        community = [s for s in reg.specs if s.provenance is Provenance.COMMUNITY]
        assert community
        assert all(not s.enabled for s in community)

    def test_local_sensors(self) -> None:
        reg = IntegrationRegistry(default_integrations())
        sensors = reg.by_kind(IntegrationKind.SENSOR)
        assert {s.name for s in sensors} == {"circadian", "resources"}
