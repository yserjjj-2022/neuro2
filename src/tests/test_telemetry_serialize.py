"""Unit tests for serialize_event — Functional Core.

Pure function: input → JSON string, no I/O, no filesystem.
"""

from __future__ import annotations

import json

import pytest

from src.telemetry.models import TelemetryEvent
from src.telemetry.serialize import serialize_event


def _event(**overrides: object) -> TelemetryEvent:
    """Собрать TelemetryEvent с дефолтами S1, переопределив нужные поля."""
    base: dict[str, object] = {
        "timestamp": 1700000000.0,
        "tick": 0,
        "free_energy": 42.5,
        "valence": -1.2,
        "allostatic_stress": 15.0,
        "gamma": 1.0,
        "active_columns": 7,
        "active_tags": "cpu",
        "reflex_tags": "",
        "bus_dim": 14,
        "latency_ms": 1.5,
        "rss_mb": 120.0,
        "drift": False,
        "memory_prior": 0.0,
        "memory_hit": False,
        "episode_stored": False,
        "phase": "phase1",
        "mode": "free",
    }
    base.update(overrides)
    return TelemetryEvent(**base)  # type: ignore[arg-type]


def test_serialize_valid(tmp_path: object) -> None:
    """Валидный event → корректная JSON-строка."""
    event = _event()
    result = serialize_event(event)
    assert isinstance(result, str)
    assert "free_energy" in result
    assert "42.5" in result
    assert "phase1" in result
    assert "free" in result


def test_serialize_all_s1_fields() -> None:
    """S2: все 18 полей присутствуют в JSON."""
    data = json.loads(
        serialize_event(_event(tick=7, gamma=2.5, active_tags="cpu,battery"))
    )
    for field in (
        "timestamp",
        "tick",
        "free_energy",
        "valence",
        "allostatic_stress",
        "gamma",
        "active_columns",
        "active_tags",
        "reflex_tags",
        "bus_dim",
        "latency_ms",
        "rss_mb",
        "drift",
        "memory_prior",
        "memory_hit",
        "episode_stored",
        "phase",
        "mode",
    ):
        assert field in data, f"missing field: {field}"
    assert data["tick"] == 7
    assert data["gamma"] == 2.5
    assert data["active_tags"] == "cpu,battery"


def test_serialize_memory_fields() -> None:
    """S2: поля памяти сериализуются корректно."""
    data = json.loads(
        serialize_event(_event(memory_prior=0.75, memory_hit=True, episode_stored=True))
    )
    assert data["memory_prior"] == 0.75
    assert data["memory_hit"] is True
    assert data["episode_stored"] is True


def test_serialize_nan_raises() -> None:
    """NaN → ValueError."""
    event = _event(timestamp=float("nan"))
    with pytest.raises(ValueError, match="NaN|Infinity"):
        serialize_event(event)


def test_serialize_infinity_raises() -> None:
    """Infinity → ValueError."""
    event = _event(timestamp=float("inf"))
    with pytest.raises(ValueError, match="NaN|Infinity"):
        serialize_event(event)


def test_serialize_negative_stress() -> None:
    """Отрицательный stress — допустимый float, не вызывает ValueError."""
    event = _event(
        allostatic_stress=-5.0, active_columns=3, phase="phase2", mode="game"
    )
    result = serialize_event(event)
    assert "-5.0" in result
    assert "phase2" in result
    assert "game" in result
