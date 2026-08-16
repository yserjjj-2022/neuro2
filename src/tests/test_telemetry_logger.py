"""Unit tests for TelemetryLogger — Shell with DI through Protocol.

Tests use mock writer — no filesystem access needed.
"""

from __future__ import annotations

import logging

import pytest

from src.telemetry.logger import TelemetryLogger
from src.telemetry.models import TelemetryEvent


class MockWriter:
    """Лёгкий mock writer для тестирования TelemetryLogger."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []
        self.write_count: int = 0

    def write(self, event: TelemetryEvent) -> None:
        self.events.append(event)
        self.write_count += 1


@pytest.fixture()
def mock_writer() -> MockWriter:
    return MockWriter()


@pytest.fixture()
def logger(mock_writer: MockWriter) -> TelemetryLogger:
    return TelemetryLogger(
        writer=mock_writer,
        phase="phase1",
        mode="free",
    )


def test_logger_with_mock_writer(logger: TelemetryLogger, mock_writer: MockWriter) -> None:
    """Logger с mock writer: проверяет вызов write."""
    logger.log(
        free_energy=42.5,
        valence=-1.2,
        allostatic_stress=15.0,
        active_columns=7,
    )

    assert mock_writer.write_count == 1
    event = mock_writer.events[0]
    assert event.free_energy == 42.5
    assert event.valence == -1.2
    assert event.allostatic_stress == 15.0
    assert event.active_columns == 7
    assert event.phase == "phase1"
    assert event.mode == "free"
    assert isinstance(event.timestamp, float)


def test_logger_swallows_writer_errors(
    logger: TelemetryLogger,
    mock_writer: MockWriter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Logger не пробрасывает исключения от writer и логирует ошибку."""
    # Заставим writer бросить исключение
    call_count = {"n": 0}

    def failing_write(event: TelemetryEvent) -> None:
        call_count["n"] += 1
        raise OSError("disk full")

    mock_writer.write = failing_write

    # Не должно проброситься
    logger.log(
        free_energy=10.0,
        valence=-0.5,
        allostatic_stress=5.0,
        active_columns=3,
    )

    # write() был вызван (исключение внутри нашей функции)
    assert call_count["n"] == 1

    # logging.error() был вызван с содержательным сообщением
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    message = caplog.records[0].message
    assert "Telemetry write failed" in message
    assert "free_energy" in message
    assert "10.0" in message


def test_logger_multiple_logs(
    logger: TelemetryLogger,
    mock_writer: MockWriter,
) -> None:
    """Несколько вызовов log() → несколько событий."""
    for i in range(5):
        logger.log(
            free_energy=float(i),
            valence=float(-i),
            allostatic_stress=float(i * 2),
            active_columns=i,
        )

    assert mock_writer.write_count == 5
    assert len(mock_writer.events) == 5


def test_logger_default_phase_mode() -> None:
    """Без явной фазы/режима — дефолтные значения."""
    mock_writer = MockWriter()
    logger = TelemetryLogger(writer=mock_writer)
    logger.log(1.0, 0.5, 0.3, 5)

    event = mock_writer.events[0]
    assert event.phase == "phase1"
    assert event.mode == "free"


def test_logger_timestamp_is_wall_clock(logger: TelemetryLogger, mock_writer: MockWriter) -> None:
    """Timestamp — wall-clock time.time(), не монотонный."""
    import time

    before = time.time()
    logger.log(1.0, 0.5, 0.3, 5)
    after = time.time()

    event = mock_writer.events[0]
    assert before <= event.timestamp <= after
