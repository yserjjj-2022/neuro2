"""Unit tests for serialize_event — Functional Core.

Pure function: input → JSON string, no I/O, no filesystem.
"""

from __future__ import annotations

import pytest

from src.telemetry.models import TelemetryEvent
from src.telemetry.serialize import serialize_event


def test_serialize_valid(tmp_path: object) -> None:
    """Валидный event → корректная JSON-строка."""
    event = TelemetryEvent(
        timestamp=1700000000.0,
        free_energy=42.5,
        valence=-1.2,
        allostatic_stress=15.0,
        active_columns=7,
        phase="phase1",
        mode="free",
    )
    result = serialize_event(event)
    assert isinstance(result, str)
    assert "free_energy" in result
    assert "42.5" in result
    assert "phase1" in result
    assert "free" in result


def test_serialize_nan_raises() -> None:
    """NaN → ValueError."""
    event = TelemetryEvent(
        timestamp=float("nan"),
        free_energy=42.5,
        valence=-1.2,
        allostatic_stress=15.0,
        active_columns=7,
        phase="phase1",
        mode="free",
    )
    with pytest.raises(ValueError, match="NaN|Infinity"):
        serialize_event(event)


def test_serialize_infinity_raises() -> None:
    """Infinity → ValueError."""
    event = TelemetryEvent(
        timestamp=float("inf"),
        free_energy=42.5,
        valence=-1.2,
        allostatic_stress=15.0,
        active_columns=7,
        phase="phase1",
        mode="free",
    )
    with pytest.raises(ValueError, match="NaN|Infinity"):
        serialize_event(event)


def test_serialize_negative_stress() -> None:
    """Отрицательный stress — допустимый float, не вызывает ValueError."""
    event = TelemetryEvent(
        timestamp=1700000000.0,
        free_energy=42.5,
        valence=-1.2,
        allostatic_stress=-5.0,
        active_columns=3,
        phase="phase2",
        mode="game",
    )
    result = serialize_event(event)
    assert "-5.0" in result
    assert "phase2" in result
    assert "game" in result
