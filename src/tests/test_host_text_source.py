"""Unit tests for TextMessageProvider — text → embedding signal."""

from __future__ import annotations

import numpy as np

from src.host.text_source import TextMessageProvider
from src.mcp import SignalCategory
from src.memory.embedder import EmbedderError, FakeEmbedder


class BrokenEmbedder:
    dim = 8

    def embed(self, text: str) -> np.ndarray:
        raise EmbedderError("down")


class TestTextMessageProvider:
    def test_dim_matches_embedder(self) -> None:
        provider = TextMessageProvider(embedder=FakeEmbedder(dim=12))
        assert provider.dim == 12

    def test_category_communicative(self) -> None:
        provider = TextMessageProvider(embedder=FakeEmbedder(dim=8))
        assert provider.category == SignalCategory.COMMUNICATIVE

    def test_text_at_boundaries(self) -> None:
        provider = TextMessageProvider(
            embedder=FakeEmbedder(dim=8),
            messages=((0, "first"), (5, "second")),
        )
        assert provider.text_at(0) == "first"
        assert provider.text_at(4) == "first"
        assert provider.text_at(5) == "second"
        assert provider.text_at(100) == "second"

    def test_text_at_no_message(self) -> None:
        provider = TextMessageProvider(embedder=FakeEmbedder(dim=8))
        assert provider.text_at(0) == ""

    def test_read_embeds_text(self) -> None:
        embedder = FakeEmbedder(dim=8)
        provider = TextMessageProvider(
            embedder=embedder, messages=((0, "hello world"),)
        )
        signal = provider.read(tick=0, now=0.0)
        assert signal.data.shape == (8,)
        np.testing.assert_array_equal(signal.data, embedder.embed("hello world"))

    def test_read_no_message_zeros(self) -> None:
        provider = TextMessageProvider(embedder=FakeEmbedder(dim=8))
        signal = provider.read(tick=0, now=0.0)
        np.testing.assert_array_equal(signal.data, np.zeros(8))

    def test_read_embedder_failure_zeros(self) -> None:
        """Сбой эмбеддера → нули, не исключение (тик не роняется)."""
        provider = TextMessageProvider(embedder=BrokenEmbedder(), messages=((0, "hi"),))
        signal = provider.read(tick=0, now=0.0)
        np.testing.assert_array_equal(signal.data, np.zeros(8))
