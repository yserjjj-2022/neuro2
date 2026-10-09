"""Integration tests for autonomy wiring (S6).

- selfcontrol observables accumulate on ticks
- consolidation is explicit and logged (invariant 6)
- ``autonomy.enabled=False`` → S5-identical contour
- telemetry autonomy fields serialized
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import AutonomyConfig, HostConfig, MemoryConfig
from src.host.loop import build_host_loop
from src.memory import Episode
from src.memory.protocols import SupportsMemory
from src.telemetry import TelemetryLogger, TelemetryWriter


def _config(tmp_path: Path, *, autonomy: AutonomyConfig) -> HostConfig:
    return HostConfig(
        log_path=str(tmp_path / "run.jsonl"),
        memory=MemoryConfig(db_path=str(tmp_path / "m.db")),
        autonomy=autonomy,
    )


class TestSelfcontrolWiring:
    def test_observables_reach_telemetry(self, tmp_path: Path) -> None:
        loop = build_host_loop(_config(tmp_path, autonomy=AutonomyConfig(enabled=True)))
        assert loop.selfcontrol is not None
        for tick in range(5):
            loop.step_once(tick)
        loop.close()
        text = (tmp_path / "run.jsonl").read_text(encoding="utf-8")
        assert "metacog_conflict" in text
        assert "metacog_metastability" in text

    def test_disabled_no_selfcontrol(self, tmp_path: Path) -> None:
        loop = build_host_loop(_config(tmp_path, autonomy=AutonomyConfig(enabled=False)))
        assert loop.selfcontrol is None
        loop.step_once(0)
        loop.close()


class TestAutonomyTelemetry:
    def test_log_accepts_autonomy_fields(self, tmp_path: Path) -> None:
        logger = TelemetryLogger(
            writer=TelemetryWriter(log_path=tmp_path / "t.jsonl")
        )
        logger.log(
            1.0,
            0.0,
            0.0,
            metacog_conflict=0.3,
            metacog_metastability=0.4,
            metacog_saturation=0.2,
            reset_level="soft",
            change_kind="development",
            consolidated_pruned=2,
        )
        text = (tmp_path / "t.jsonl").read_text(encoding="utf-8")
        assert "metacog_conflict" in text
        assert '"reset_level": "soft"' in text
        assert '"consolidated_pruned": 2' in text

    def test_defaults_backward_compat(self, tmp_path: Path) -> None:
        logger = TelemetryLogger(
            writer=TelemetryWriter(log_path=tmp_path / "t.jsonl")
        )
        logger.log(1.0, 0.0, 0.0)
        assert (tmp_path / "t.jsonl").exists()


class TestConsolidationWiring:
    def test_consolidate_memory_prunes_and_logs(self, tmp_path: Path) -> None:
        autonomy = AutonomyConfig(
            enabled=True,
            consolidate_min_weight=1.0,
            recency_tau_s=1.0,
            schema_threshold=0.9,
            max_schemas=4,
        )
        loop = build_host_loop(_config(tmp_path, autonomy=autonomy))
        assert loop.memory is not None
        store: SupportsMemory = loop.memory.store
        store.store(
            Episode(
                content="important",
                embedding=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                timestamp=1000.0,
                valence=10.0,
                stress=1.0,
                free_energy=5.0,
            )
        )
        store.store(
            Episode(
                content="trivial",
                embedding=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                timestamp=0.0,
                valence=0.0,
                stress=0.0,
                free_energy=0.0,
            )
        )
        # consolidate_memory использует self.clock() (wall) — фиксируем now.
        loop.clock = lambda: 1000.0
        pruned = loop.consolidate_memory()
        assert pruned == 1
        loop.step_once(0)
        loop.close()
        text = (tmp_path / "run.jsonl").read_text(encoding="utf-8")
        assert '"consolidated_pruned": 1' in text

    def test_consolidate_disabled_memory_returns_zero(self, tmp_path: Path) -> None:
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(enabled=False),
            autonomy=AutonomyConfig(enabled=True),
        )
        loop = build_host_loop(config)
        assert loop.consolidate_memory() == 0
        loop.close()


class TestNightCycle:
    """Ночной цикл: консолидация по расписанию/объёму (S6 проход 2)."""

    def test_night_cycle_triggers_and_logs(self, tmp_path: Path) -> None:
        autonomy = AutonomyConfig(
            enabled=True,
            consolidate_every_ticks=5,
            consolidate_min_episodes=1,
            consolidate_min_weight=1.0,
            recency_tau_s=1.0,
        )
        loop = build_host_loop(_config(tmp_path, autonomy=autonomy))
        assert loop.memory is not None
        store: SupportsMemory = loop.memory.store
        store.store(
            Episode(
                content="trivial",
                embedding=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                timestamp=0.0,
                valence=0.0,
                stress=0.0,
                free_energy=0.0,
            )
        )
        # Synthetic-время: now = tick * dt. Порог веса 1.0 → старый эпизод
        # удаляется (affect=0, свежесть мала).
        for tick in range(6):
            loop.step_once(tick)
        contents = [e.content for e in store.all_episodes()]
        loop.close()
        text = (tmp_path / "run.jsonl").read_text(encoding="utf-8")
        assert '"consolidated_pruned": 1' in text
        assert "trivial" not in contents

    def test_night_cycle_disabled_by_default(self, tmp_path: Path) -> None:
        autonomy = AutonomyConfig(enabled=True)
        loop = build_host_loop(_config(tmp_path, autonomy=autonomy))
        assert loop.memory is not None
        loop.memory.store.store(
            Episode(
                content="trivial",
                embedding=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                timestamp=0.0,
                valence=0.0,
                stress=0.0,
                free_energy=0.0,
            )
        )
        for tick in range(20):
            loop.step_once(tick)
        contents = [e.content for e in loop.memory.store.all_episodes()]
        loop.close()
        assert "trivial" in contents


class TestS5Compatibility:
    def test_disabled_contour_identical(self, tmp_path: Path) -> None:
        """autonomy.enabled=False → selfcontrol None, контекст без метакогниции."""
        loop = build_host_loop(_config(tmp_path, autonomy=AutonomyConfig(enabled=False)))
        loop.step_once(0)
        ctx = loop.policy_context(has_new_message=False)
        assert ctx.metacognition is None
        loop.close()
