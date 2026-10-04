"""Unit tests for memory consolidation (S6): pruning + schemas."""

from __future__ import annotations

import numpy as np
import pytest

from src.memory import (
    Episode,
    MemoryStore,
    consolidate,
    episode_weight,
    plan_consolidation,
    should_consolidate,
)


def _episode(
    ep_id: int,
    *,
    embedding: np.ndarray,
    timestamp: float = 0.0,
    valence: float = 0.0,
    stress: float = 0.0,
    content: str = "x",
) -> Episode:
    return Episode(
        content=content,
        embedding=embedding,
        timestamp=timestamp,
        valence=valence,
        stress=stress,
        free_energy=1.0,
        id=ep_id,
    )


class TestEpisodeWeight:
    def test_fresh_high_affect_heavier(self) -> None:
        fresh = _episode(1, embedding=np.ones(4), timestamp=100.0, valence=5.0)
        old = _episode(2, embedding=np.ones(4), timestamp=0.0, valence=5.0)
        w_fresh = episode_weight(fresh, now=100.0, recency_tau_s=10.0)
        w_old = episode_weight(old, now=100.0, recency_tau_s=10.0)
        assert w_fresh > w_old

    def test_bad_tau_raises(self) -> None:
        with pytest.raises(ValueError):
            episode_weight(_episode(1, embedding=np.ones(2)), now=0.0, recency_tau_s=0.0)


class TestPlanConsolidation:
    def test_prunes_low_weight(self) -> None:
        episodes = [
            _episode(1, embedding=np.ones(4), timestamp=100.0, valence=10.0),
            _episode(2, embedding=np.ones(4), timestamp=0.0, valence=0.0),
        ]
        plan = plan_consolidation(
            episodes, min_weight=1.0, schema_threshold=0.9, max_schemas=4,
            now=100.0, recency_tau_s=1.0,
        )
        assert 2 in plan.prune_ids
        assert 1 not in plan.prune_ids
        assert plan.kept == 1

    def test_schemas_from_close_episodes(self) -> None:
        base = np.array([1.0, 0.0, 0.0, 0.0])
        episodes = [
            _episode(i, embedding=base + 0.001 * i, timestamp=float(i), valence=10.0)
            for i in range(1, 4)
        ]
        plan = plan_consolidation(
            episodes, min_weight=0.0, schema_threshold=0.9, max_schemas=4,
            now=10.0, recency_tau_s=1e9,
        )
        assert len(plan.schemas) == 1
        assert plan.schemas[0].member_count == 3

    def test_distinct_episodes_separate_schemas(self) -> None:
        episodes = [
            _episode(1, embedding=np.array([1.0, 0.0, 0.0]), valence=10.0),
            _episode(2, embedding=np.array([0.0, 1.0, 0.0]), valence=10.0),
        ]
        plan = plan_consolidation(
            episodes, min_weight=0.0, schema_threshold=0.9, max_schemas=4,
            now=0.0, recency_tau_s=1e9,
        )
        assert len(plan.schemas) == 2

    def test_deterministic(self) -> None:
        episodes = [
            _episode(i, embedding=np.array([1.0, 0.1 * i, 0.0]), valence=1.0)
            for i in range(1, 5)
        ]
        a = plan_consolidation(
            episodes, min_weight=0.0, schema_threshold=0.9,
            max_schemas=4, now=0.0, recency_tau_s=1e9,
        )
        b = plan_consolidation(
            episodes, min_weight=0.0, schema_threshold=0.9,
            max_schemas=4, now=0.0, recency_tau_s=1e9,
        )
        assert a.prune_ids == b.prune_ids
        assert len(a.schemas) == len(b.schemas)

    def test_max_schemas_zero(self) -> None:
        episodes = [_episode(1, embedding=np.array([1.0, 0.0]), valence=10.0)]
        plan = plan_consolidation(
            episodes, min_weight=0.0, schema_threshold=0.9, max_schemas=0,
            now=0.0, recency_tau_s=1e9,
        )
        assert plan.schemas == ()

    def test_bad_params(self) -> None:
        with pytest.raises(ValueError):
            plan_consolidation([], min_weight=-1.0, schema_threshold=0.9,
                               max_schemas=1, now=0.0)
        with pytest.raises(ValueError):
            plan_consolidation([], min_weight=0.0, schema_threshold=2.0,
                               max_schemas=1, now=0.0)


class TestConsolidateShell:
    def test_prunes_and_saves(self, tmp_path) -> None:
        store = MemoryStore(db_path=tmp_path / "m.db", embedding_dim=4)
        # значимый свежий эпизод
        store.store(
            Episode(
                content="important",
                embedding=np.array([1.0, 0.0, 0.0, 0.0]),
                timestamp=100.0,
                valence=10.0,
                stress=1.0,
                free_energy=5.0,
            )
        )
        # незначимый старый эпизод
        store.store(
            Episode(
                content="trivial",
                embedding=np.array([0.0, 1.0, 0.0, 0.0]),
                timestamp=0.0,
                valence=0.0,
                stress=0.0,
                free_energy=0.0,
            )
        )
        result = consolidate(
            store, min_weight=1.0, schema_threshold=0.9, max_schemas=4,
            now=100.0, recency_tau_s=1.0,
        )
        assert result.pruned == 1
        assert result.schemas_saved >= 1
        remaining = store.all_episodes()
        assert len(remaining) == 1
        assert remaining[0].content == "important"
        store.close()

    def test_delete_empty_noop(self, tmp_path) -> None:
        store = MemoryStore(db_path=tmp_path / "m.db", embedding_dim=2)
        assert store.delete([]) == 0
        store.close()

    def test_schema_saved_visible(self, tmp_path) -> None:
        store = MemoryStore(db_path=tmp_path / "m.db", embedding_dim=2)
        schema_id = store.save_schema(np.array([1.0, 0.0]), 3, "summary")
        assert schema_id >= 1
        store.close()


class TestConsolidationTrigger:
    """Ночной цикл: чистое решение о запуске (S6 проход 2)."""

    def test_disabled_when_zero_interval(self) -> None:
        trigger = should_consolidate(
            tick=100, last_tick=0, episode_count=50,
            every_ticks=0, min_episodes=1,
        )
        assert not trigger.due

    def test_not_due_before_interval(self) -> None:
        trigger = should_consolidate(
            tick=10, last_tick=0, episode_count=50,
            every_ticks=100, min_episodes=1,
        )
        assert not trigger.due
        assert "interval" in trigger.reason

    def test_not_due_without_enough_episodes(self) -> None:
        trigger = should_consolidate(
            tick=100, last_tick=0, episode_count=2,
            every_ticks=100, min_episodes=10,
        )
        assert not trigger.due
        assert "episodes" in trigger.reason

    def test_due_when_interval_and_volume_reached(self) -> None:
        trigger = should_consolidate(
            tick=100, last_tick=0, episode_count=10,
            every_ticks=100, min_episodes=10,
        )
        assert trigger.due

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            should_consolidate(
                tick=-1, last_tick=0, episode_count=0,
                every_ticks=1, min_episodes=0,
            )
        with pytest.raises(ValueError):
            should_consolidate(
                tick=1, last_tick=0, episode_count=0,
                every_ticks=1, min_episodes=-1,
            )
