"""Unit tests for TelemetryWriter — Imperative Shell.

Tests file I/O without mocking — uses real filesystem via tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.telemetry.models import TelemetryEvent
from src.telemetry.writer import TelemetryWriter


@pytest.fixture()
def sample_event() -> TelemetryEvent:
    return TelemetryEvent(
        timestamp=1700000000.0,
        free_energy=42.5,
        valence=-1.2,
        allostatic_stress=15.0,
        active_columns=7,
        phase="phase1",
        mode="free",
    )


def test_write_creates_file(tmp_path: Path, sample_event: TelemetryEvent) -> None:
    """Writer создаёт/обновляет файл при вызове write()."""
    log_file = tmp_path / "telemetry_new.jsonl"
    assert not log_file.exists()
    writer = TelemetryWriter(log_file)
    writer.write(sample_event)
    assert log_file.exists()
    # Файл содержит ровно одну строку JSON
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["free_energy"] == 42.5
    writer.close()


def test_write_jsonl_format(tmp_path: Path, sample_event: TelemetryEvent) -> None:
    """Каждая строка — валидный JSON."""
    log_file = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(log_file)

    writer.write(sample_event)
    writer.write(
        TelemetryEvent(
            timestamp=1700000001.0,
            free_energy=43.0,
            valence=-1.3,
            allostatic_stress=16.0,
            active_columns=8,
            phase="phase1",
            mode="free",
        )
    )
    writer.close()

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    for line in lines:
        data = json.loads(line)
        assert "free_energy" in data
        assert "phase" in data
        assert "mode" in data


def test_write_append_mode(tmp_path: Path, sample_event: TelemetryEvent) -> None:
    """Writer дозаписывает в существующий файл (append)."""
    log_file = tmp_path / "telemetry.jsonl"
    # Предварительно запишем что-то
    log_file.write_text('{"existing": true}\n', encoding="utf-8")

    writer = TelemetryWriter(log_file)
    writer.write(sample_event)
    writer.close()

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["existing"] is True


def test_write_closed_raises(tmp_path: Path, sample_event: TelemetryEvent) -> None:
    """После close() write() выбрасывает RuntimeError."""
    log_file = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(log_file)
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.write(sample_event)


def test_close_twice_no_error(tmp_path: Path) -> None:
    """Двойной close() не выбрасывает исключение."""
    log_file = tmp_path / "telemetry.jsonl"
    writer = TelemetryWriter(log_file)
    writer.close()
    writer.close()  # should not raise
