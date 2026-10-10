"""Tests for TOML override loader (ADR-0011 §4): base + override, fail-fast."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.integrations.models import IntegrationKind, Provenance, StdioTransport
from src.integrations.toml_loader import load_integrations


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "integrations.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadIntegrations:
    def test_base_only(self) -> None:
        reg = load_integrations()
        assert reg.find("weather") is not None

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_integrations(tmp_path / "nope.toml")

    def test_override_existing(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [[integrations]]
            name = "weather"
            kind = "tool"
            transport = "stdio"
            command = "npx"
            args = ["-y", "custom-weather"]
            enabled = false
            """,
        )
        reg = load_integrations(path)
        weather = reg.find("weather")
        assert weather is not None
        assert weather.enabled is False
        assert isinstance(weather.transport, StdioTransport)
        assert weather.transport.args == ("-y", "custom-weather")

    def test_tool_args(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [[integrations]]
            name = "time"
            enabled = true
            tool_args = { get_current_time = { timezone = "Europe/Moscow" } }
            """,
        )
        reg = load_integrations(path)
        time = reg.find("time")
        assert time is not None
        assert time.args_for("get_current_time") == {"timezone": "Europe/Moscow"}
        assert time.args_for("convert_time") == {}

    def test_tool_args_bad_type_fail_fast(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [[integrations]]
            name = "news"
            kind = "tool"
            transport = "http"
            url = "https://example.com/mcp"
            tool_args = "oops"
            """,
        )
        with pytest.raises(TypeError):
            load_integrations(path)

    def test_partial_override_keeps_transport(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [[integrations]]
            name = "everything"
            enabled = false
            """,
        )
        reg = load_integrations(path)
        everything = reg.find("everything")
        assert everything is not None
        assert everything.enabled is False
        assert isinstance(everything.transport, StdioTransport)

    def test_append_new(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [[integrations]]
            name = "news"
            kind = "tool"
            transport = "http"
            url = "https://example.com/mcp"
            provenance = "community"
            """,
        )
        reg = load_integrations(path)
        assert reg.find("news") is not None

    def test_unknown_key_fail_fast(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [[integrations]]
            name = "x"
            transport = "stdio"
            command = "npx"
            bogus = 1
            """,
        )
        with pytest.raises(ValueError):
            load_integrations(path)

    def test_unknown_kind_fail_fast(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [[integrations]]
            name = "x"
            kind = "banana"
            transport = "stdio"
            command = "npx"
            """,
        )
        with pytest.raises(ValueError):
            load_integrations(path)

    def test_unknown_transport_fail_fast(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [[integrations]]
            name = "x"
            transport = "carrier-pigeon"
            """,
        )
        with pytest.raises(ValueError):
            load_integrations(path)

    def test_unknown_top_level_fail_fast(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "bogus = 1\n")
        with pytest.raises(ValueError):
            load_integrations(path)

    def test_stdio_requires_command(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [[integrations]]
            name = "x"
            transport = "stdio"
            """,
        )
        with pytest.raises(ValueError):
            load_integrations(path)

    def test_local_provenance(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [[integrations]]
            name = "circadian"
            kind = "sensor"
            transport = "local"
            provider = "circadian"
            provenance = "local"
            """,
        )
        reg = load_integrations(path)
        spec = reg.find("circadian")
        assert spec is not None
        assert spec.kind is IntegrationKind.SENSOR
        assert spec.provenance is Provenance.LOCAL

    def test_rank_override(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [[integrations]]
            name = "battery"
            rank = 3.0
            """,
        )
        reg = load_integrations(path)
        battery = reg.find("battery")
        assert battery is not None
        assert battery.rank == pytest.approx(3.0)

    def test_rank_absent_is_none(self, tmp_path: Path) -> None:
        """Без rank важность не объявлена (legacy F = 0.5·Σγ·e²)."""
        reg = load_integrations()
        battery = reg.find("battery")
        assert battery is not None
        assert battery.rank is None

    def test_rank_non_positive_fail_fast(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [[integrations]]
            name = "battery"
            rank = 0.0
            """,
        )
        with pytest.raises(ValueError):
            load_integrations(path)

    def test_rank_bad_type_fail_fast(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
            [[integrations]]
            name = "battery"
            rank = "high"
            """,
        )
        with pytest.raises(TypeError):
            load_integrations(path)
