"""Integration tests for the host loop (src/host/loop.py).

Verifies the full experimental path end-to-end with real file I/O:
    SignalBus → CMCPipeline → telemetry JSONL

These tests encode the experimental protocol (see plan):
    #1 constant input → F → 0
    #2 step input → F spike + negative valence
    #3 battery drain → reflex signal
    #4 noise → stress accumulates
    #5 default bus → N ticks → N JSONL events
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import HostConfig
from src.core.cmc import ColumnConfig
from src.host.loop import HostLoop, build_host_loop
from src.host.sources import (
    BatteryProvider,
    ConstantProvider,
    NoisyProvider,
    SignalBus,
    StepProvider,
)
from src.host.wiring import build_cmc_pipeline


def _loop(tmp_path: Path, providers: list, k: int = 1, dt: float = 0.0) -> HostLoop:
    """Собрать loop под произвольный набор провайдеров.

    TaskAttractor требует ≥ 2 колонок (нужен runner-up), поэтому ансамбль
    всегда из двух колонок с одинаковым входом.
    """
    bus = SignalBus(providers)
    columns = [
        ColumnConfig(input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="a"),
        ColumnConfig(input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="b"),
    ]
    pipeline = build_cmc_pipeline(
        columns=columns,
        k=k,
        log_path=tmp_path / "run.jsonl",
        active_threshold=1e-8,
    )
    return HostLoop(bus=bus, pipeline=pipeline, dt=dt, clock=lambda: 0.0)


def _read_events(tmp_path: Path) -> list[dict]:
    """Прочитать JSONL телеметрии."""
    path = tmp_path / "run.jsonl"
    return [json.loads(line) for line in path.read_text().strip().split("\n")]


class TestHostLoopExperiment:
    """Экспериментальный протокол: запуск → метрики → вывод."""

    def test_exp1_constant_input_converges(self, tmp_path: Path) -> None:
        """#1: стабильный вход → F → 0, active → 0 (EMA сходится)."""
        loop = _loop(tmp_path, [ConstantProvider(value=(1.0, 2.0))])
        for tick in range(200):
            result = loop.step_once(tick)
        loop.close()

        assert result.f < 1e-6
        assert loop.pipeline.ensemble.active == 0

    def test_exp2_step_input_causes_spike(self, tmp_path: Path) -> None:
        """#2: скачок входа → F растёт, valence < 0, затем сходимость."""
        loop = _loop(
            tmp_path,
            [StepProvider(before=(1.0,), after=(10.0,), step_at=50)],
        )
        # Сходимся на before (тики 0..49), f_before — последний тик до скачка
        for tick in range(50):
            result = loop.step_once(tick)
        f_before = result.f

        # Скачок на тике 50: F должна подскочить, valence — уйти в минус
        spike = loop.step_once(50)
        assert spike.f > f_before
        assert spike.valence < 0.0

        # Затем снова сходимся
        for tick in range(51, 300):
            result = loop.step_once(tick)
        loop.close()
        assert result.f < f_before

    def test_exp3_battery_reflex(self, tmp_path: Path) -> None:
        """#3: разряд батареи → сигнал помечен is_reflex при severity ≥ 0.9."""
        provider = BatteryProvider(start_level=1.0, drain_per_tick=0.01)
        loop = _loop(tmp_path, [provider])

        for tick in range(95):
            loop.step_once(tick)
        loop.close()

        battery = next(s for s in loop.bus.last_signals if s.tag == "battery")
        assert battery.severity >= 0.9
        assert battery.is_reflex is True

    def test_exp4_noise_accumulates_stress(self, tmp_path: Path) -> None:
        """#4: белый шум → allostatic_stress накапливается."""
        loop = _loop(tmp_path, [NoisyProvider(dim=3, scale=1.0, seed=0)])
        stresses = [loop.step_once(tick).allostatic_stress for tick in range(100)]
        loop.close()

        assert stresses[-1] > stresses[0]
        assert stresses[-1] > stresses[50]

    def test_exp5_default_bus_n_ticks(self, tmp_path: Path) -> None:
        """#5: дефолтный набор → N тиков → N событий в JSONL."""
        config = HostConfig(dt=0.0, log_path=str(tmp_path / "run.jsonl"))
        loop = build_host_loop(config)
        executed = loop.run(20)
        loop.close()

        assert executed == 20
        events = _read_events(tmp_path)
        assert len(events) == 20
        assert loop.bus.bus_dim == 12
        assert events[0]["active_columns"] == 3


class TestHostLoopMechanics:
    """Механика loop: precision, валидация, graceful shutdown."""

    def test_precision_shape(self, tmp_path: Path) -> None:
        """precision = ones(N_columns * bus_dim)."""
        loop = _loop(tmp_path, [ConstantProvider(value=(1.0, 2.0))])
        precision = loop.precision()
        assert precision.shape == (2 * 2,)
        assert np.all(precision == 1.0)

    def test_precision_shape_default(self, tmp_path: Path) -> None:
        config = HostConfig(dt=0.0, log_path=str(tmp_path / "run.jsonl"))
        loop = build_host_loop(config)
        assert loop.precision().shape == (3 * 12,)

    def test_variance_mode_not_implemented(self, tmp_path: Path) -> None:
        """precision_mode='variance' — задел Фазы 2, явный NotImplementedError."""
        config = HostConfig(
            dt=0.0,
            precision_mode="variance",
            log_path=str(tmp_path / "run.jsonl"),
        )
        loop = build_host_loop(config)
        with pytest.raises(NotImplementedError):
            loop.precision()

    def test_invalid_precision_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            HostConfig(precision_mode="bogus")

    def test_run_zero_ticks(self, tmp_path: Path) -> None:
        loop = _loop(tmp_path, [ConstantProvider(value=(1.0,))])
        assert loop.run(0) == 0
        loop.close()

    def test_negative_max_ticks_raises(self, tmp_path: Path) -> None:
        loop = _loop(tmp_path, [ConstantProvider(value=(1.0,))])
        with pytest.raises(ValueError):
            loop.run(-1)

    def test_negative_dt_raises(self, tmp_path: Path) -> None:
        bus = SignalBus([ConstantProvider(value=(1.0,))])
        pipeline = build_cmc_pipeline(
            columns=[
                ColumnConfig(input_dim=1, state_dim=1),
                ColumnConfig(input_dim=1, state_dim=1),
            ],
            k=1,
            log_path=tmp_path / "run.jsonl",
        )
        with pytest.raises(ValueError):
            HostLoop(bus=bus, pipeline=pipeline, dt=-1.0)

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        loop = _loop(tmp_path, [ConstantProvider(value=(1.0,))])
        loop.step_once(0)
        loop.close()
        loop.close()  # не должно бросить

    def test_deterministic_across_loops(self, tmp_path: Path) -> None:
        """Одинаковые провайдеры и зерно → одинаковая телеметрия."""
        loop_a = _loop(tmp_path / "a", [ConstantProvider(value=(1.0, 2.0))])
        loop_b = _loop(tmp_path / "b", [ConstantProvider(value=(1.0, 2.0))])
        for tick in range(10):
            fa = loop_a.step_once(tick)
            fb = loop_b.step_once(tick)
        loop_a.close()
        loop_b.close()
        assert fa.f == fb.f
