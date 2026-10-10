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

from src.config import HostConfig, MemoryConfig
from src.core.cmc import ColumnConfig
from src.core.energy import DriftDetector, PrecisionEstimator
from src.host.loop import HostLoop, build_host_loop
from src.host.resources import ResourceMeter
from src.host.sources import (
    BatteryProvider,
    ConstantProvider,
    NoisyProvider,
    SignalBus,
    StepProvider,
)
from src.host.wiring import build_cmc_pipeline
from src.mcp import SignalCategory, SignalSource
from src.telemetry import TelemetryLogger, TelemetryWriter


def _loop(
    tmp_path: Path,
    providers: list,
    k: int = 1,
    tick_dt: float = 0.01,
    precision_mode: str = "ones",
    f_threshold: float = 1e9,
    stress_threshold: float = 1e9,
) -> HostLoop:
    """Собрать полный HostLoop под произвольный набор провайдеров.

    TaskAttractor требует ≥ 2 колонок (нужен runner-up), поэтому ансамбль
    всегда из двух колонок. precision_mode по умолчанию "ones" для
    предсказуемости тестов.
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
    writer = TelemetryWriter(log_path=tmp_path / "run.jsonl")
    logger = TelemetryLogger(writer=writer, phase="phase1", mode="free")
    return HostLoop(
        bus=bus,
        pipeline=pipeline,
        logger=logger,
        estimator=PrecisionEstimator(dim=bus.bus_dim, window=50),
        meter=ResourceMeter(),
        drift=DriftDetector(
            f_threshold=f_threshold,
            stress_threshold=stress_threshold,
        ),
        tick_dt=tick_dt,
        clock_mode="synthetic",
        precision_mode=precision_mode,
    )


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
            outcome = loop.step_once(tick)
        loop.close()

        assert outcome.result.f < 1e-6
        assert loop.pipeline.ensemble.active == 0

    def test_exp2_step_input_causes_spike(self, tmp_path: Path) -> None:
        """#2: скачок входа → F растёт, valence < 0, затем сходимость."""
        loop = _loop(
            tmp_path,
            [StepProvider(before=(1.0,), after=(10.0,), step_at=50)],
        )
        # Сходимся на before (тики 0..49), f_before — последний тик до скачка
        for tick in range(50):
            outcome = loop.step_once(tick)
        f_before = outcome.result.f

        # Скачок на тике 50: F должна подскочить, valence — уйти в минус
        spike = loop.step_once(50)
        assert spike.result.f > f_before
        assert spike.result.valence < 0.0

        # Затем снова сходимся
        for tick in range(51, 300):
            outcome = loop.step_once(tick)
        loop.close()
        assert outcome.result.f < f_before

    def test_exp3_battery_reflex_visible_in_log(self, tmp_path: Path) -> None:
        """#3: разряд батареи → reflex виден в телеметрии (reflex_tags)."""
        provider = BatteryProvider(start_level=1.0, drain_per_tick=0.01)
        loop = _loop(tmp_path, [provider])

        for tick in range(95):
            loop.step_once(tick)
        loop.close()

        battery = next(s for s in loop.bus.last_signals if s.tag == "battery")
        assert battery.severity >= 0.9
        assert battery.is_reflex is True

        events = _read_events(tmp_path)
        assert "battery" in events[-1]["reflex_tags"]

    def test_exp4_noise_accumulates_stress(self, tmp_path: Path) -> None:
        """#4: белый шум → allostatic_stress накапливается."""
        loop = _loop(tmp_path, [NoisyProvider(dim=3, scale=1.0, seed=0)])
        stresses = [
            loop.step_once(tick).result.allostatic_stress for tick in range(100)
        ]
        loop.close()

        assert stresses[-1] > stresses[0]
        assert stresses[-1] > stresses[50]

    def test_exp5_default_bus_n_ticks(self, tmp_path: Path) -> None:
        """#5: дефолтный набор → N тиков → N событий в JSONL."""
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "mem.db")),
        )
        loop = build_host_loop(config)
        executed = loop.run(20)
        loop.close()

        assert executed == 20
        events = _read_events(tmp_path)
        assert len(events) == 20
        assert loop.bus.bus_dim == 14  # +2 ресурсных канала
        assert loop.total_dim == 18  # +4 приор памяти
        assert events[0]["active_columns"] == 3
        assert events[0]["tick"] == 0
        assert events[19]["tick"] == 19


class TestChannelImportance:
    """Веса каналов (BACKLOG): rank₀ → per-component w, вклад в F."""

    def test_no_ranks_disables_importance(self, tmp_path: Path) -> None:
        """Без channel_ranks importance=None (legacy F = 0.5·Σγ·e²)."""
        config = HostConfig(log_path=str(tmp_path / "run.jsonl"))
        loop = build_host_loop(config)
        loop.close()
        assert loop.importance is None

    def test_ranks_produce_weight_vector(self, tmp_path: Path) -> None:
        """С channel_ranks importance = rank/dim по компонентам."""
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            channel_ranks=(("battery", 3.0),),
        )
        loop = build_host_loop(config)
        loop.close()
        assert loop.importance is not None
        assert loop.importance.shape == (loop.total_dim,)
        # battery dim=1 → вес 3.0; прочие каналы rank=1 → 1/dim
        battery_seg = next(s for s in loop.bus.segments if s.name == "battery")
        assert loop.importance[battery_seg.offset] == pytest.approx(3.0)

    def test_rank_scales_channel_contribution(self, tmp_path: Path) -> None:
        """rank повышает вклад канала в F и виден в channel_contrib."""
        base = HostConfig(log_path=str(tmp_path / "a.jsonl"))
        loop_base = build_host_loop(base)
        for tick in range(20):
            outcome_base = loop_base.step_once(tick)
        loop_base.close()

        boosted = HostConfig(
            log_path=str(tmp_path / "b.jsonl"),
            channel_ranks=(("battery", 5.0),),
        )
        loop_boost = build_host_loop(boosted)
        for tick in range(20):
            outcome_boost = loop_boost.step_once(tick)
        loop_boost.close()

        contrib_base = dict(outcome_base.channel_contrib)
        contrib_boost = dict(outcome_boost.channel_contrib)
        assert "battery" in contrib_base
        assert contrib_boost["battery"] > contrib_base["battery"]

    def test_channel_contrib_logged(self, tmp_path: Path) -> None:
        """channel_contrib попадает в телеметрию (наблюдаемость весов)."""
        config = HostConfig(log_path=str(tmp_path / "run.jsonl"))
        loop = build_host_loop(config)
        loop.run(5)
        loop.close()

        events = _read_events(tmp_path)
        assert events[0]["channel_contrib"] != ""
        assert "battery:" in events[0]["channel_contrib"]

    def test_rank_from_registry_enables_importance(self, tmp_path: Path) -> None:
        """rank₀ из записи реестра включает веса (BACKLOG-целевое место)."""
        from src.integrations import IntegrationRegistry
        from src.integrations.models import (
            IntegrationKind,
            IntegrationSpec,
            LocalTransport,
            Provenance,
        )

        registry = IntegrationRegistry(
            (
                IntegrationSpec(
                    name="battery",
                    kind=IntegrationKind.SENSOR,
                    transport=LocalTransport("battery"),
                    provenance=Provenance.LOCAL,
                    rank=3.0,
                ),
            )
        )
        config = HostConfig(log_path=str(tmp_path / "run.jsonl"))
        loop = build_host_loop(config, integrations=registry)
        loop.close()
        assert loop.importance is not None
        assert loop.importance.shape == (loop.total_dim,)
        battery_seg = next(s for s in loop.bus.segments if s.name == "battery")
        assert loop.importance[battery_seg.offset] == pytest.approx(3.0)

    def test_registry_subset_narrows_bus(self, tmp_path: Path) -> None:
        """Подмножество SENSOR в реестре → узкая шина (масштабируемость)."""
        from src.integrations import IntegrationRegistry
        from src.integrations.models import (
            IntegrationKind,
            IntegrationSpec,
            LocalTransport,
            Provenance,
        )

        registry = IntegrationRegistry(
            (
                IntegrationSpec(
                    name="circadian",
                    kind=IntegrationKind.SENSOR,
                    transport=LocalTransport("circadian"),
                    provenance=Provenance.LOCAL,
                ),
                IntegrationSpec(
                    name="battery",
                    kind=IntegrationKind.SENSOR,
                    transport=LocalTransport("battery"),
                    provenance=Provenance.LOCAL,
                ),
            )
        )
        config = HostConfig(log_path=str(tmp_path / "run.jsonl"))
        loop = build_host_loop(config, integrations=registry)
        loop.close()
        assert [s.name for s in loop.bus.segments] == ["circadian", "battery"]
        assert loop.bus.bus_dim == 3  # 2 (circadian) + 1 (battery)

    def test_channel_ranks_override_wins(self, tmp_path: Path) -> None:
        """HostConfig.channel_ranks — override поверх ранга реестра."""
        from src.integrations import IntegrationRegistry
        from src.integrations.models import (
            IntegrationKind,
            IntegrationSpec,
            LocalTransport,
            Provenance,
        )

        registry = IntegrationRegistry(
            (
                IntegrationSpec(
                    name="battery",
                    kind=IntegrationKind.SENSOR,
                    transport=LocalTransport("battery"),
                    provenance=Provenance.LOCAL,
                    rank=2.0,
                ),
            )
        )
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            channel_ranks=(("battery", 5.0),),
        )
        loop = build_host_loop(config, integrations=registry)
        loop.close()
        battery_seg = next(s for s in loop.bus.segments if s.name == "battery")
        assert loop.importance is not None
        assert loop.importance[battery_seg.offset] == pytest.approx(5.0)


class TestHostLoopMechanics:
    """Механика loop: precision, время, валидация, graceful shutdown."""

    def test_recall_only_on_new_message(self, tmp_path: Path) -> None:
        """Событийность: recall запускается только при новом тексте.

        Тики частые, сообщения редкие. Пока текст не меняется, recall
        (SQLite MATCH) не вызывается — иначе он дёргался бы каждый тик.
        """
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "mem.db")),
        )
        loop = build_host_loop(
            config,
            meter=ResourceMeter(),
            messages=((0, "hello"), (20, "hello")),  # тот же текст дважды
        )
        recall_calls = {"n": 0}
        original = loop.memory.recall_prior  # type: ignore[union-attr]

        def counting_recall(query: object) -> object:
            recall_calls["n"] += 1
            return original(query)  # type: ignore[arg-type]

        loop.memory.recall_prior = counting_recall  # type: ignore[union-attr,method-assign]

        loop.run(40)
        loop.close()
        # 40 тиков, но recall только на смене текста (тик 0); тик 20 — тот же.
        assert recall_calls["n"] == 1

    def test_sensory_io_excluded_from_latency(self, tmp_path: Path) -> None:
        """Медленный сенсорный провайдер не влияет на ресурсную латентность.

        Реальный API-эмбеддер медленнее бюджета тика; если бы его время
        попадало в latency, severity ≥ 0.9 давал бы постоянный reflex.
        Loop измеряет только вычислительную фазу.
        """
        import time as _time

        class SlowProvider:
            tag = "slow"
            category = SignalCategory.EXTEROCEPTIVE
            dim = 1
            period = 1

            def read(self, tick: int, now: float) -> SignalSource:
                _time.sleep(0.02)  # 20 мс — сетевой I/O
                return SignalSource(
                    category=self.category,
                    data=np.array([1.0]),
                    tag=self.tag,
                )

        loop = _loop(tmp_path, [SlowProvider()], tick_dt=0.01)
        loop.step_once(0)
        loop.close()

        # Сон был 20 мс, но latency (compute) много меньше
        assert loop.meter.last_latency_s < 0.01
        assert not any(s.is_reflex for s in loop.bus.last_signals)

    def test_precision_ones_shape(self, tmp_path: Path) -> None:
        loop = _loop(tmp_path, [ConstantProvider(value=(1.0, 2.0))])
        u = np.array([1.0, 2.0])
        precision = loop.precision(u)
        assert precision.shape == (2 * 2,)
        assert np.all(precision == 1.0)

    def test_precision_variance_shape(self, tmp_path: Path) -> None:
        loop = _loop(
            tmp_path,
            [ConstantProvider(value=(1.0, 2.0))],
            precision_mode="variance",
        )
        u = np.array([1.0, 2.0])
        assert loop.precision(u).shape == (2 * 2,)

    def test_precision_shape_default(self, tmp_path: Path) -> None:
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "mem.db")),
        )
        loop = build_host_loop(config)
        u = np.zeros(loop.total_dim)
        assert loop.precision(u).shape == (3 * 18,)

    def test_invalid_precision_mode_raises(self) -> None:
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

    def test_invalid_tick_dt_raises(self, tmp_path: Path) -> None:
        bus = SignalBus([ConstantProvider(value=(1.0,))])
        pipeline = build_cmc_pipeline(
            columns=[
                ColumnConfig(input_dim=1, state_dim=1),
                ColumnConfig(input_dim=1, state_dim=1),
            ],
            k=1,
            log_path=tmp_path / "run.jsonl",
        )
        writer = TelemetryWriter(log_path=tmp_path / "run.jsonl")
        logger = TelemetryLogger(writer=writer)
        with pytest.raises(ValueError):
            HostLoop(
                bus=bus,
                pipeline=pipeline,
                logger=logger,
                estimator=PrecisionEstimator(dim=1),
                meter=ResourceMeter(),
                drift=DriftDetector(f_threshold=1.0, stress_threshold=1.0),
                tick_dt=0.0,
            )

    def test_invalid_clock_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            HostConfig(clock_mode="bogus")

    def test_synthetic_clock_deterministic(self, tmp_path: Path) -> None:
        """synthetic: dt = tick_dt, now = tick·tick_dt."""
        loop = _loop(tmp_path, [ConstantProvider(value=(1.0,))], tick_dt=0.02)
        dt, now = loop._time_for_tick(5)
        assert dt == pytest.approx(0.02)
        assert now == pytest.approx(0.1)

    def test_wall_clock_positive_dt(self, tmp_path: Path) -> None:
        """wall: dt измеряется, всегда > 0."""
        loop = _loop(tmp_path, [ConstantProvider(value=(1.0,))])
        loop.clock_mode = "wall"
        times = iter([100.0, 100.5, 100.9])
        loop.clock = lambda: next(times)
        dt1, _ = loop._time_for_tick(0)
        dt2, _ = loop._time_for_tick(1)
        assert dt1 == pytest.approx(0.01)  # первый тик → tick_dt
        assert dt2 == pytest.approx(0.5)

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        loop = _loop(tmp_path, [ConstantProvider(value=(1.0,))])
        loop.step_once(0)
        loop.close()
        loop.close()  # не должно бросить

    def test_deterministic_across_loops(self, tmp_path: Path) -> None:
        """Одинаковые провайдеры → одинаковая телеметрия (synthetic)."""
        loop_a = _loop(tmp_path / "a", [ConstantProvider(value=(1.0, 2.0))])
        loop_b = _loop(tmp_path / "b", [ConstantProvider(value=(1.0, 2.0))])
        for tick in range(10):
            oa = loop_a.step_once(tick)
            ob = loop_b.step_once(tick)
        loop_a.close()
        loop_b.close()
        assert oa.result.f == ob.result.f
