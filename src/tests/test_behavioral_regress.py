"""Behavioral regression — canonical scenarios and organism invariants.

VALIDATION.md levels: unit/integration prove functions; these tests prove the
ORGANISM — that the whole loop behaves sanely on scripted scenarios. We assert
invariants (VALIDATION.md §2), not exact values.

Scenarios C1–C6 (VALIDATION.md §3). ``behavioral_fingerprint`` is the compact
summary used for regression comparison across code changes.
"""

from __future__ import annotations

import itertools
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
from src.telemetry import TelemetryLogger, TelemetryWriter


def behavioral_fingerprint(
    events: list[dict],
    valence_significance: float = 1.0,
) -> dict[str, float]:
    """Компактная сводка прогона для регресса (VALIDATION.md §5).

    Чистая функция: не мутирует events.

    Args:
        events: Список событий телеметрии (dict из JSONL).
        valence_significance: Порог |valence|, ниже которого колебания
            считаются микро-шумом у нуля (не «сменой настроения»).

    Returns:
        Словарь метрик: профиль F, стресс, valence, reflex, drift, latency.
    """
    if not events:
        raise ValueError("events must not be empty")

    f_values = [e["free_energy"] for e in events]
    stress_values = [e["allostatic_stress"] for e in events]
    valence_values = [e["valence"] for e in events]
    latency_values = [e["latency_ms"] for e in events]

    # Значимые смены знака — только среди |valence| > порога
    significant = [v for v in valence_values if abs(v) > valence_significance]
    sign_changes = sum(
        1 for a, b in itertools.pairwise(significant) if (a > 0) != (b > 0)
    )
    reflex_events = sum(1 for e in events if e["reflex_tags"])

    return {
        "n_events": float(len(events)),
        "f_min": float(min(f_values)),
        "f_max": float(max(f_values)),
        "f_final": float(f_values[-1]),
        "stress_max": float(max(stress_values)),
        "stress_final": float(stress_values[-1]),
        "valence_sign_changes": float(sign_changes),
        "reflex_events": float(reflex_events),
        "drift_events": float(sum(1 for e in events if e["drift"])),
        "memory_hits": float(sum(1 for e in events if e.get("memory_hit"))),
        "episodes_stored": float(sum(1 for e in events if e.get("episode_stored"))),
        "throttle_events": float(sum(1 for e in events if e.get("throttle"))),
        "policy_events": float(
            sum(1 for e in events if e.get("policy_action", "") != "")
        ),
        "latency_p50_ms": float(np.percentile(latency_values, 50)),
        "latency_p95_ms": float(np.percentile(latency_values, 95)),
    }


class FakeMeter(ResourceMeter):
    """Детерминированный meter для replay-тестов (реальный RSS нестабилен)."""

    def __init__(self, latency_s: float = 0.0002, rss_mb: float = 100.0) -> None:
        super().__init__()
        self._fake_latency = latency_s
        self._fake_rss = rss_mb

    @property
    def last_latency_s(self) -> float:
        return self._fake_latency

    @property
    def last_rss_mb(self) -> float:
        return self._fake_rss

    def record_tick(self, latency_s: float) -> None:
        pass


def _loop(
    tmp_path: Path,
    providers: list,
    precision_mode: str = "variance",
    f_threshold: float = 1e9,
    stress_threshold: float = 1e9,
) -> HostLoop:
    """Собрать HostLoop под сценарий (2 колонки, synthetic время, fake meter)."""
    bus = SignalBus(providers)
    columns = [
        ColumnConfig(input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="a"),
        ColumnConfig(input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="b"),
    ]
    pipeline = build_cmc_pipeline(
        columns=columns, k=1, log_path=tmp_path / "run.jsonl", active_threshold=1e-8
    )
    writer = TelemetryWriter(log_path=tmp_path / "run.jsonl")
    logger = TelemetryLogger(writer=writer, phase="phase1", mode="free")
    return HostLoop(
        bus=bus,
        pipeline=pipeline,
        logger=logger,
        estimator=PrecisionEstimator(dim=bus.bus_dim, window=50),
        meter=FakeMeter(),
        drift=DriftDetector(f_threshold=f_threshold, stress_threshold=stress_threshold),
        tick_dt=0.1,
        clock_mode="synthetic",
        precision_mode=precision_mode,
    )


def _events(tmp_path: Path) -> list[dict]:
    path = tmp_path / "run.jsonl"
    return [json.loads(line) for line in path.read_text().strip().split("\n")]


def _assert_no_nan(events: list[dict]) -> None:
    """Инвариант организма §2.4: нет NaN/inf в телеметрии."""
    for e in events:
        for field in ("free_energy", "valence", "allostatic_stress", "gamma"):
            assert np.isfinite(e[field]), f"non-finite {field}: {e[field]}"


class TestCanonicalScenarios:
    """C1–C6: канонические сценарии (VALIDATION.md §3)."""

    def test_c1_constant_converges(self, tmp_path: Path) -> None:
        """C1: стабильный вход → F→0, без NaN."""
        loop = _loop(tmp_path, [ConstantProvider(value=(1.0, 2.0))])
        for tick in range(200):
            loop.step_once(tick)
        loop.close()
        events = _events(tmp_path)

        fp = behavioral_fingerprint(events)
        assert fp["f_final"] < 1e-6
        _assert_no_nan(events)

    def test_c2_step_spike_then_recovery(self, tmp_path: Path) -> None:
        """C2: скачок → F↑, valence<0, затем сходимость."""
        loop = _loop(tmp_path, [StepProvider(before=(1.0,), after=(10.0,), step_at=50)])
        for tick in range(50):
            outcome = loop.step_once(tick)
        f_before = outcome.result.f
        spike = loop.step_once(50)
        for tick in range(51, 300):
            loop.step_once(tick)
        loop.close()

        assert spike.result.f > f_before
        assert spike.result.valence < 0.0
        _assert_no_nan(_events(tmp_path))

    def test_c3_battery_reflex_in_log(self, tmp_path: Path) -> None:
        """C3: критический сигнал → reflex виден в телеметрии."""
        loop = _loop(tmp_path, [BatteryProvider(start_level=1.0, drain_per_tick=0.01)])
        for tick in range(95):
            loop.step_once(tick)
        loop.close()
        events = _events(tmp_path)

        fp = behavioral_fingerprint(events)
        assert fp["reflex_events"] >= 1
        assert "battery" in events[-1]["reflex_tags"]

    def test_c4_noise_stress_accumulates_no_runaway(self, tmp_path: Path) -> None:
        """C4: шум → стресс копится, но без runaway (инвариант ограниченности)."""
        loop = _loop(tmp_path, [NoisyProvider(dim=3, scale=1.0, seed=0)])
        for tick in range(100):
            loop.step_once(tick)
        loop.close()
        events = _events(tmp_path)

        fp = behavioral_fingerprint(events)
        assert fp["stress_final"] > 0.0
        assert np.isfinite(fp["stress_max"])
        _assert_no_nan(events)

    def test_c5_default_bus_stability(self, tmp_path: Path) -> None:
        """C5: дефолтный набор → N тиков → N событий, без shape mismatch."""
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "mem.db")),
        )
        loop = build_host_loop(config)
        executed = loop.run(50)
        loop.close()
        events = _events(tmp_path)

        assert executed == 50
        assert len(events) == 50
        _assert_no_nan(events)

    def test_c6_resource_signal_in_log(self, tmp_path: Path) -> None:
        """C6: ресурсный сигнал измеряется и логируется (latency/rss)."""
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "mem.db")),
        )
        loop = build_host_loop(config, meter=FakeMeter())
        loop.run(20)
        loop.close()
        events = _events(tmp_path)

        assert all(
            "resources" in e["active_tags"] or e["active_tags"] == "" for e in events
        )
        assert all(e["latency_ms"] >= 0.0 for e in events)
        assert all(e["rss_mb"] > 0.0 for e in events)
        assert all(e["bus_dim"] == 14 for e in events)

    def test_c11_memory_store_and_recall(self, tmp_path: Path) -> None:
        """C11 (S2): эпизоды пишутся на значимых событиях; recall находит.

        Сценарий: стабильный текст + скачок входа → всплеск F → эпизод
        записан. Затем повторный прогон на той же БД → recall находит эпизод
        (memory_hit) по тому же тексту.
        """
        db = tmp_path / "mem.db"
        messages = ((0, "the cat sat on the mat"),)
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(db), episode_spike_threshold=0.5),
        )
        loop = build_host_loop(config, meter=FakeMeter(), messages=messages)
        loop.run(200)
        loop.close()

        events = _events(tmp_path)
        assert any(e["episode_stored"] for e in events), "no episode stored on spike"

        # Повторный прогон на той же БД: recall должен найти эпизод
        log2 = tmp_path / "run2.jsonl"
        config2 = HostConfig(
            log_path=str(log2),
            memory=MemoryConfig(db_path=str(db), episode_spike_threshold=0.5),
        )
        loop2 = build_host_loop(config2, meter=FakeMeter(), messages=messages)
        loop2.run(200)
        loop2.close()
        events2 = [json.loads(line) for line in log2.read_text().strip().split("\n")]
        assert any(e["memory_hit"] for e in events2), "recall found nothing"


class TestLongHorizon:
    """C10: длинный горизонт (VALIDATION.md §3, S6 проход 2).

    Проверяем устойчивость организма на длинном прогоне: отсутствие тихого
    дрейфа, ограниченность, невырождение активности и детерминизм. Значения
    эталона — в VALIDATION.md §4 (S6, C10); тест утверждает ИНВАРИАНТЫ, а не
    точные числа (VALIDATION.md §5).
    """

    def _run(self, tmp_path: Path, ticks: int = 1500) -> list[dict]:
        """Длинный прогон с автономией и памятью (детерминированный)."""
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(
                db_path=str(tmp_path / "mem.db"), embedder_mode="fake"
            ),
        )
        loop = build_host_loop(config, meter=FakeMeter())
        loop.run(ticks)
        loop.close()
        return _events(tmp_path)

    def test_c10_no_silent_drift(self, tmp_path: Path) -> None:
        """C10: длинный горизонт без тихого дрейфа (инвариант §2.11)."""
        events = self._run(tmp_path)
        fp = behavioral_fingerprint(events)

        # Тихого дрейфа нет: детектор не срабатывает за 1500 тиков.
        assert fp["drift_events"] == 0
        # Классификатор не деградирует: только stable/development, не drift.
        kinds = {e["change_kind"] for e in events}
        assert kinds <= {"", "stable", "development"}, kinds
        _assert_no_nan(events)

    def test_c10_bounded_and_active(self, tmp_path: Path) -> None:
        """C10: ограниченность и невырождение активности на длинном горизонте."""
        events = self._run(tmp_path)
        fp = behavioral_fingerprint(events)

        assert np.isfinite(fp["f_max"]) and fp["f_max"] < 1e3
        assert np.isfinite(fp["stress_max"]) and fp["stress_max"] < 1e3
        # Активность не коллапсирует: нет длинной серии тиков без активных колонок.
        longest = _longest_zero_streak([e["active_columns"] for e in events])
        assert longest <= 10, f"activity collapse: {longest} zero-active ticks"

    def test_c10_determinism_long_horizon(self, tmp_path: Path) -> None:
        """C10: одинаковый seed → одинаковый отпечаток на длинном горизонте."""
        events_a = self._run(tmp_path / "a")
        events_b = self._run(tmp_path / "b")
        fp_a = behavioral_fingerprint(events_a)
        fp_b = behavioral_fingerprint(events_b)

        assert fp_a == fp_b


def _longest_zero_streak(values: list[int]) -> int:
    """Длина самой длинной серии нулевой активности (чистая)."""
    best = current = 0
    for value in values:
        current = current + 1 if value == 0 else 0
        best = max(best, current)
    return best


class TestOrganismInvariants:
    """Инварианты организма (VALIDATION.md §2), применимые к S1."""

    def test_valence_no_jitter_on_stable_input(self, tmp_path: Path) -> None:
        """§2.2: на стабильном входе valence сглажена (не дребезжит).

        Дребезг — это микро-колебания у нуля при стабильном входе. На белом
        шуме колебания F легитимны (вход реально меняется каждый тик).
        """
        loop = _loop(tmp_path, [ConstantProvider(value=(1.0, 2.0))])
        for tick in range(1000):
            loop.step_once(tick)
        loop.close()
        events = _events(tmp_path)

        fp = behavioral_fingerprint(events)
        # До S1 было ~649; сглаживание на стабильном входе → единицы
        assert fp["valence_sign_changes"] <= 5

    def test_determinism_synthetic(self, tmp_path: Path) -> None:
        """§2.8: synthetic clock + fake meter + fresh DB → replay."""
        config_a = HostConfig(
            log_path=str(tmp_path / "a" / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "a" / "mem.db")),
        )
        loop_a = build_host_loop(config_a, meter=FakeMeter())
        loop_a.run(30)
        loop_a.close()

        config_b = HostConfig(
            log_path=str(tmp_path / "b" / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "b" / "mem.db")),
        )
        loop_b = build_host_loop(config_b, meter=FakeMeter())
        loop_b.run(30)
        loop_b.close()

        fp_a = behavioral_fingerprint(_events(tmp_path / "a"))
        fp_b = behavioral_fingerprint(_events(tmp_path / "b"))
        assert fp_a["f_final"] == pytest.approx(fp_b["f_final"])
        assert fp_a["valence_sign_changes"] == fp_b["valence_sign_changes"]

    def test_fingerprint_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            behavioral_fingerprint([])

    def test_fingerprint_keys(self, tmp_path: Path) -> None:
        loop = _loop(tmp_path, [ConstantProvider(value=(1.0,))])
        loop.run(5)
        loop.close()
        fp = behavioral_fingerprint(_events(tmp_path))
        assert "f_final" in fp
        assert "reflex_events" in fp
        assert "latency_p95_ms" in fp


class TestS4Gates:
    """Ворота S4 (VALIDATION.md §4): goal-directed, reflex, explainability."""

    def test_goal_directed_via_full_loop(self, tmp_path: Path) -> None:
        """Goal-directed: смена Preferences меняет действие без переобучения."""
        from src.core.policy import Action, Preferences, select_action

        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "mem.db")),
        )
        loop = build_host_loop(config, meter=FakeMeter())
        loop.run(3)
        ctx = loop.policy_context(has_new_message=True)
        loop.close()

        assert select_action(ctx, Preferences()).chosen is Action.RESPOND
        assert (
            select_action(ctx, Preferences(respond_to_messages=False)).chosen
            is not Action.RESPOND
        )

    def test_explainability_trace_present(self, tmp_path: Path) -> None:
        """Explainability: у решения есть непустая причинная трасса."""
        from src.core.policy import Preferences, select_action

        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "mem.db")),
        )
        loop = build_host_loop(config, meter=FakeMeter())
        loop.run(3)
        trace = select_action(loop.policy_context(has_new_message=True), Preferences())
        loop.close()

        assert trace.reason != ""
        assert len(trace.candidates) == 5
        winner = next(c for c in trace.candidates if c.action is trace.chosen)
        assert winner.value == max(c.value for c in trace.candidates)

    def test_reflex_gate_in_full_loop(self, tmp_path: Path) -> None:
        """Reflex: критический сигнал → throttle в телеметрии (≤ 1 тик)."""
        from src.core.homeostasis import Setpoint

        provider = BatteryProvider(start_level=1.0, drain_per_tick=0.01)
        bus = SignalBus([provider])
        columns = [
            ColumnConfig(
                input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="a"
            ),
            ColumnConfig(
                input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="b"
            ),
        ]
        pipeline = build_cmc_pipeline(
            columns=columns, k=1, log_path=tmp_path / "run.jsonl"
        )
        writer = TelemetryWriter(log_path=tmp_path / "run.jsonl")
        from src.core.homeostasis import Homeostat

        loop = HostLoop(
            bus=bus,
            pipeline=pipeline,
            logger=TelemetryLogger(writer=writer, phase="phase1", mode="free"),
            estimator=PrecisionEstimator(dim=bus.bus_dim, window=50),
            meter=FakeMeter(),
            drift=DriftDetector(f_threshold=1e9, stress_threshold=1e9),
            homeostat=Homeostat(
                setpoints=(Setpoint(tag="battery", comfort=0.5, critical=0.9),)
            ),
            tick_dt=0.1,
            clock_mode="synthetic",
            precision_mode="ones",
        )
        for tick in range(91):
            loop.step_once(tick)
        loop.close()

        events = _events(tmp_path)
        fp = behavioral_fingerprint(events)
        assert fp["throttle_events"] >= 1
        assert events[-1]["throttle"] is True
