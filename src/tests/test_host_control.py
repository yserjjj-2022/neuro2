"""Unit tests for ControlChannel — hot-testing commands over HostLoop (S4).

Minimal S4 set: status/pause/resume/step. No I/O: the channel is driven
directly in tests (loop with synthetic clock).
"""

from __future__ import annotations

from pathlib import Path

from src.core.cmc import ColumnConfig
from src.core.energy import DriftDetector, PrecisionEstimator
from src.host.control import ControlChannel
from src.host.loop import HostLoop
from src.host.resources import ResourceMeter
from src.host.sources import ConstantProvider, SignalBus
from src.host.wiring import build_cmc_pipeline
from src.telemetry import TelemetryLogger, TelemetryWriter


def _loop(tmp_path: Path) -> HostLoop:
    """Собрать loop с синтетическим временем (2 колонки)."""
    bus = SignalBus([ConstantProvider(value=(1.0,))])
    columns = [
        ColumnConfig(input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="a"),
        ColumnConfig(input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="b"),
    ]
    pipeline = build_cmc_pipeline(columns=columns, k=1, log_path=tmp_path / "run.jsonl")
    writer = TelemetryWriter(log_path=tmp_path / "run.jsonl")
    return HostLoop(
        bus=bus,
        pipeline=pipeline,
        logger=TelemetryLogger(writer=writer, phase="phase1", mode="free"),
        estimator=PrecisionEstimator(dim=bus.bus_dim, window=50),
        meter=ResourceMeter(),
        drift=DriftDetector(f_threshold=1e9, stress_threshold=1e9),
        clock_mode="synthetic",
        precision_mode="ones",
    )


class TestControlChannel:
    """Tests for status/pause/resume/step."""

    def test_step_makes_exact_ticks(self, tmp_path: Path) -> None:
        """step(n) делает ровно n тиков."""
        channel = ControlChannel(loop=_loop(tmp_path))
        assert channel.step(3) == 3
        assert channel.tick == 3
        assert channel.steps == 3
        channel.loop.close()

    def test_step_zero_noop(self, tmp_path: Path) -> None:
        """step(0) не двигает loop."""
        channel = ControlChannel(loop=_loop(tmp_path))
        assert channel.step(0) == 0
        assert channel.tick == 0
        channel.loop.close()

    def test_step_negative_raises(self, tmp_path: Path) -> None:
        """step(-1) → ValueError."""
        channel = ControlChannel(loop=_loop(tmp_path))
        try:
            channel.step(-1)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("step(-1) should raise")
        channel.loop.close()

    def test_pause_blocks_run(self, tmp_path: Path) -> None:
        """pause → run не делает тиков."""
        channel = ControlChannel(loop=_loop(tmp_path))
        channel.pause()
        assert channel.paused is True
        assert channel.run(max_ticks=10) == 0
        assert channel.tick == 0
        channel.loop.close()

    def test_step_works_while_paused(self, tmp_path: Path) -> None:
        """step работает на паузе (ручной тик)."""
        channel = ControlChannel(loop=_loop(tmp_path))
        channel.pause()
        assert channel.step(2) == 2
        assert channel.tick == 2
        channel.loop.close()

    def test_resume_allows_run(self, tmp_path: Path) -> None:
        """resume после pause → run снова делает тики."""
        channel = ControlChannel(loop=_loop(tmp_path))
        channel.pause()
        channel.resume()
        assert channel.paused is False
        assert channel.run(max_ticks=4) == 4
        channel.loop.close()

    def test_status_before_ticks(self, tmp_path: Path) -> None:
        """status до тиков не падает и даёт строку состояния."""
        channel = ControlChannel(loop=_loop(tmp_path))
        line = channel.status()
        assert line.startswith("[F=")
        channel.loop.close()

    def test_status_after_ticks(self, tmp_path: Path) -> None:
        """status после тика содержит поля состояния."""
        channel = ControlChannel(loop=_loop(tmp_path))
        channel.step(1)
        line = channel.status()
        assert "задача=" in line
        assert "дрейф=" in line
        channel.loop.close()
