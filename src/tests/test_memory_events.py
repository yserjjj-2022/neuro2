"""Unit tests for events + prior — pure core, no SQLite."""

from __future__ import annotations

import numpy as np
import pytest

from src.memory.events import build_event_content, is_significant_event
from src.memory.models import Episode
from src.memory.prior import MEMORY_PRIOR_DIM, encode_memory_prior


class TestIsSignificantEvent:
    def test_reflex_is_significant(self) -> None:
        assert is_significant_event(0.0, 0.0, ("battery",), 1.0) is True

    def test_spike_above_threshold(self) -> None:
        assert is_significant_event(5.0, 1.0, (), 1.0) is True

    def test_spike_at_threshold_not_significant(self) -> None:
        """Строго больше порога (не >=)."""
        assert is_significant_event(2.0, 1.0, (), 1.0) is False

    def test_flat_not_significant(self) -> None:
        assert is_significant_event(1.0, 1.0, (), 1.0) is False

    def test_drop_not_significant(self) -> None:
        assert is_significant_event(0.5, 2.0, (), 1.0) is False

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            is_significant_event(1.0, 0.0, (), -1.0)


class TestBuildEventContent:
    def test_nonempty_and_deterministic(self) -> None:
        a = build_event_content(("cpu",), ("battery",), -0.5, 2.0)
        b = build_event_content(("cpu",), ("battery",), -0.5, 2.0)
        assert a == b
        assert a != ""
        assert "cpu" in a and "battery" in a

    def test_empty_tags_placeholder(self) -> None:
        content = build_event_content((), (), 0.0, 0.0)
        assert "none" in content


class TestEncodeMemoryPrior:
    def _episode(self, embedding: list[float]) -> Episode:
        return Episode(
            content="x",
            embedding=np.array(embedding, dtype=np.float64),
            timestamp=0.0,
            valence=1.0,
            stress=2.0,
            free_energy=3.0,
        )

    def test_none_episode_zero_prior(self) -> None:
        prior = encode_memory_prior(None, np.ones(4))
        np.testing.assert_array_equal(prior, np.zeros(MEMORY_PRIOR_DIM))

    def test_none_query_zero_prior(self) -> None:
        prior = encode_memory_prior(self._episode([1.0, 0.0, 0.0, 0.0]), None)
        np.testing.assert_array_equal(prior, np.zeros(MEMORY_PRIOR_DIM))

    def test_shape_and_bounds(self) -> None:
        ep = self._episode([1.0, 0.0, 0.0, 0.0])
        prior = encode_memory_prior(ep, np.array([1.0, 0.0, 0.0, 0.0]))
        assert prior.shape == (MEMORY_PRIOR_DIM,)
        assert np.all(prior >= -1.0) and np.all(prior <= 1.0)
        assert np.all(np.isfinite(prior))

    def test_cosine_component(self) -> None:
        """Идентичные векторы → cos == 1.0 в первой компоненте."""
        ep = self._episode([1.0, 2.0, 3.0, 4.0])
        prior = encode_memory_prior(ep, np.array([1.0, 2.0, 3.0, 4.0]))
        assert prior[0] == pytest.approx(1.0)

    def test_affect_tanh_bounded(self) -> None:
        """Большие аффекты не выходят за пределы tanh."""
        ep = Episode(
            content="x",
            embedding=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=0.0,
            valence=100.0,
            stress=100.0,
            free_energy=100.0,
        )
        prior = encode_memory_prior(ep, np.array([1.0, 0.0, 0.0, 0.0]))
        assert prior[1] <= 1.0 and prior[2] <= 1.0 and prior[3] <= 1.0
        assert prior[1] > 0.0

    def test_zero_query_norm_zero_cosine(self) -> None:
        ep = self._episode([1.0, 0.0, 0.0, 0.0])
        prior = encode_memory_prior(ep, np.zeros(4))
        assert prior[0] == 0.0

    def test_shape_mismatch_raises(self) -> None:
        ep = self._episode([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            encode_memory_prior(ep, np.ones(3))
