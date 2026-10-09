"""Unit tests for MemoryRouter — shell orchestration with fake store/embedder."""

from __future__ import annotations

import numpy as np
import pytest

from src.memory.embedder import EmbedderError, FakeEmbedder
from src.memory.errors import MemoryStoreError
from src.memory.models import Episode
from src.memory.prior import MEMORY_PRIOR_DIM
from src.memory.router import MemoryRouter


class FakeStore:
    """Минимальный in-memory store для тестов router (без SQLite)."""

    def __init__(self) -> None:
        self.episodes: list[Episode] = []
        self.fail_store = False
        self.fail_recall = False

    def store(self, episode: Episode) -> int:
        if self.fail_store:
            raise MemoryStoreError("boom")
        self.episodes.append(episode)
        return len(self.episodes)

    def recall(self, query: np.ndarray, limit: int = 5) -> list[Episode]:
        if self.fail_recall:
            raise MemoryStoreError("boom")
        return self.episodes[:limit]

    def all_episodes(self) -> list[Episode]:
        return list(self.episodes)

    def delete(self, ids: list[int]) -> int:
        return len(ids)

    def save_schema(
        self, centroid: np.ndarray, member_count: int, summary: str
    ) -> int:
        return member_count

    def count(self) -> int:
        return len(self.episodes)


def _router(
    store: FakeStore | None = None,
    spike_threshold: float = 1.0,
    dim: int = 8,
) -> MemoryRouter:
    return MemoryRouter(
        store=store or FakeStore(),
        embedder=FakeEmbedder(dim=dim),
        spike_threshold=spike_threshold,
    )


class TestContextEmbedding:
    def test_empty_text_none(self) -> None:
        assert _router().context_embedding("") is None

    def test_embeds_text(self) -> None:
        emb = _router().context_embedding("hello world")
        assert emb is not None and emb.shape == (8,)

    def test_cache_avoids_recompute(self) -> None:
        router = _router()
        first = router.context_embedding("same text")
        second = router.context_embedding("same text")
        assert first is second


class TestRecallPrior:
    def test_none_query_zero_prior(self) -> None:
        prior = _router().recall_prior(None)
        np.testing.assert_array_equal(prior, np.zeros(MEMORY_PRIOR_DIM))

    def test_empty_store_zero_prior(self) -> None:
        prior = _router().recall_prior(np.ones(8))
        np.testing.assert_array_equal(prior, np.zeros(MEMORY_PRIOR_DIM))

    def test_recall_returns_prior(self) -> None:
        store = FakeStore()
        router = _router(store=store)
        router.maybe_store(
            text="the cat sat",
            query=router.context_embedding("the cat sat"),
            f=10.0,
            prev_f=0.0,
            valence=1.0,
            stress=1.0,
            active_tags=(),
            reflex_tags=(),
            now=0.0,
        )
        prior = router.recall_prior(router.context_embedding("the cat sat"))
        assert prior.shape == (MEMORY_PRIOR_DIM,)
        assert prior[0] == pytest.approx(1.0)

    def test_recall_failure_returns_zeros(self) -> None:
        store = FakeStore()
        store.fail_recall = True
        prior = _router(store=store).recall_prior(np.ones(8))
        np.testing.assert_array_equal(prior, np.zeros(MEMORY_PRIOR_DIM))


class TestMaybeStore:
    def test_insignificant_not_stored(self) -> None:
        store = FakeStore()
        router = _router(store=store)
        result = router.maybe_store(
            text="",
            query=None,
            f=1.0,
            prev_f=1.0,
            valence=0.0,
            stress=0.0,
            active_tags=(),
            reflex_tags=(),
            now=0.0,
        )
        assert result is None
        assert store.episodes == []

    def test_spike_stored(self) -> None:
        store = FakeStore()
        router = _router(store=store)
        result = router.maybe_store(
            text="",
            query=None,
            f=10.0,
            prev_f=0.0,
            valence=-1.0,
            stress=2.0,
            active_tags=("cpu",),
            reflex_tags=(),
            now=0.0,
        )
        assert result == 1
        assert len(store.episodes) == 1

    def test_reflex_stored(self) -> None:
        store = FakeStore()
        router = _router(store=store)
        result = router.maybe_store(
            text="",
            query=None,
            f=0.0,
            prev_f=0.0,
            valence=0.0,
            stress=0.0,
            active_tags=(),
            reflex_tags=("battery",),
            now=0.0,
        )
        assert result == 1

    def test_uses_text_as_content(self) -> None:
        store = FakeStore()
        router = _router(store=store)
        router.maybe_store(
            text="operator said hi",
            query=router.context_embedding("operator said hi"),
            f=10.0,
            prev_f=0.0,
            valence=0.0,
            stress=0.0,
            active_tags=(),
            reflex_tags=(),
            now=0.0,
        )
        assert store.episodes[0].content == "operator said hi"

    def test_store_failure_returns_none(self) -> None:
        store = FakeStore()
        store.fail_store = True
        router = _router(store=store)
        result = router.maybe_store(
            text="",
            query=None,
            f=10.0,
            prev_f=0.0,
            valence=0.0,
            stress=0.0,
            active_tags=(),
            reflex_tags=(),
            now=0.0,
        )
        assert result is None

    def test_embed_failure_returns_none(self) -> None:
        class BrokenEmbedder:
            dim = 8

            def embed(self, text: str) -> np.ndarray:
                raise EmbedderError("down")

        router = MemoryRouter(
            store=FakeStore(), embedder=BrokenEmbedder(), spike_threshold=1.0
        )
        result = router.maybe_store(
            text="",
            query=None,
            f=10.0,
            prev_f=0.0,
            valence=0.0,
            stress=0.0,
            active_tags=(),
            reflex_tags=(),
            now=0.0,
        )
        assert result is None


class TestValidation:
    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryRouter(FakeStore(), FakeEmbedder(8), spike_threshold=-1.0)

    def test_bad_recall_limit_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryRouter(FakeStore(), FakeEmbedder(8), 1.0, recall_limit=0)

    def test_bad_prior_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryRouter(FakeStore(), FakeEmbedder(8), 1.0, prior_dim=0)
