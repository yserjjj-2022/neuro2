"""Unit tests for embedder — pure core (FakeEmbedder) + factory."""

from __future__ import annotations

import numpy as np
import pytest

from src.memory.embedder import (
    ApiEmbedder,
    Embedder,
    FakeEmbedder,
    build_embedder,
)


class TestFakeEmbedder:
    def test_deterministic(self) -> None:
        """Одинаковый текст → одинаковый вектор (replay)."""
        emb = FakeEmbedder(dim=16)
        a = emb.embed("hello world")
        b = emb.embed("hello world")
        np.testing.assert_array_equal(a, b)

    def test_shape_and_finite(self) -> None:
        emb = FakeEmbedder(dim=8)
        vec = emb.embed("some text here")
        assert vec.shape == (8,)
        assert np.all(np.isfinite(vec))

    def test_unit_norm(self) -> None:
        """Непустой текст → L2-норма == 1."""
        emb = FakeEmbedder(dim=16)
        vec = emb.embed("normalized vector test")
        assert np.linalg.norm(vec) == pytest.approx(1.0)

    def test_empty_text_zero_vector(self) -> None:
        """Пустой текст → нули (не ошибка, не NaN)."""
        emb = FakeEmbedder(dim=8)
        vec = emb.embed("")
        np.testing.assert_array_equal(vec, np.zeros(8))
        assert emb.embed("   ").tolist() == [0.0] * 8

    def test_similar_texts_higher_cosine(self) -> None:
        """Общие токены → выше косинус, чем у несвязанного текста."""
        emb = FakeEmbedder(dim=64)
        base = emb.embed("the cat sat on the mat")
        similar = emb.embed("the cat sat on the rug")
        unrelated = emb.embed("quantum chromodynamics lagrangian")
        cos_similar = float(np.dot(base, similar))
        cos_unrelated = float(np.dot(base, unrelated))
        assert cos_similar > cos_unrelated

    def test_identical_text_cosine_one(self) -> None:
        emb = FakeEmbedder(dim=32)
        vec = emb.embed("repeatable phrase")
        assert float(np.dot(vec, vec)) == pytest.approx(1.0)

    def test_rejects_invalid_dim(self) -> None:
        with pytest.raises(ValueError):
            FakeEmbedder(dim=0)

    def test_is_embedder(self) -> None:
        assert isinstance(FakeEmbedder(dim=4), Embedder)


class TestBuildEmbedder:
    def test_fake_mode(self) -> None:
        emb = build_embedder(mode="fake", dim=12)
        assert isinstance(emb, FakeEmbedder)
        assert emb.dim == 12

    def test_auto_without_key_is_fake(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        emb = build_embedder(mode="auto", dim=10)
        assert isinstance(emb, FakeEmbedder)
        assert emb.dim == 10

    def test_auto_with_key_is_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        emb = build_embedder(mode="auto")
        assert isinstance(emb, ApiEmbedder)

    def test_api_without_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            build_embedder(mode="api")

    def test_api_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        emb = build_embedder(mode="api", api_key="sk-explicit")
        assert isinstance(emb, ApiEmbedder)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            build_embedder(mode="nonsense")


class TestApiEmbedderContract:
    def test_no_key_embed_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Без ключа embed() → EmbedderError (ленивый клиент)."""
        from src.memory.embedder import EmbedderError

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        emb = ApiEmbedder()
        with pytest.raises(EmbedderError):
            emb.embed("hello")

    def test_rejects_invalid_dim(self) -> None:
        with pytest.raises(ValueError):
            ApiEmbedder(dim=0)
