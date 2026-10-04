"""Unit tests for integration registry models (ADR-0011): Core data + validation."""

from __future__ import annotations

import dataclasses

import pytest

from src.integrations.models import (
    HttpTransport,
    IntegrationKind,
    IntegrationSpec,
    LocalTransport,
    Provenance,
    StdioTransport,
)
from src.mcp.models import SignalCategory


class TestTransports:
    def test_stdio_requires_command(self) -> None:
        with pytest.raises(ValueError):
            StdioTransport("")

    def test_http_requires_url(self) -> None:
        with pytest.raises(ValueError):
            HttpTransport("")

    def test_local_requires_provider(self) -> None:
        with pytest.raises(ValueError):
            LocalTransport("")

    def test_stdio_ok(self) -> None:
        t = StdioTransport("npx", ("-y", "pkg"), (("KEY", "v"),))
        assert t.command == "npx"
        assert t.args == ("-y", "pkg")
        assert t.env == (("KEY", "v"),)


class TestIntegrationSpec:
    def _spec(self, **overrides: object) -> IntegrationSpec:
        base: dict[str, object] = {
            "name": "weather",
            "kind": IntegrationKind.TOOL,
            "transport": StdioTransport("npx", ("-y", "@dangahagan/weather-mcp")),
            "category": SignalCategory.EXTEROCEPTIVE,
            "provenance": Provenance.OFFICIAL,
        }
        base.update(overrides)
        return IntegrationSpec(**base)  # type: ignore[arg-type]

    def test_valid(self) -> None:
        spec = self._spec()
        assert spec.is_mcp
        assert spec.enabled
        assert spec.reversible

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._spec(name="")

    def test_bad_period_rejected(self) -> None:
        with pytest.raises(ValueError):
            self._spec(period=0)

    def test_local_transport_only_for_sensor(self) -> None:
        with pytest.raises(ValueError):
            self._spec(
                kind=IntegrationKind.TOOL,
                transport=LocalTransport("circadian"),
            )

    def test_local_sensor_ok(self) -> None:
        spec = self._spec(
            name="circadian",
            kind=IntegrationKind.SENSOR,
            transport=LocalTransport("circadian"),
            provenance=Provenance.LOCAL,
        )
        assert not spec.is_mcp

    def test_frozen(self) -> None:
        spec = self._spec()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.name = "x"  # type: ignore[misc]
