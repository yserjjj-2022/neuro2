"""Unit tests for embedder — pure core (FakeEmbedder) + factory."""

from __future__ import annotations

import numpy as np
import pytest

from src.memory.embedder import (
    DEFAULT_API_DIM,
    DEFAULT_BASE_URL,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_DIM,
    ENV_MODEL,
    ApiEmbedder,
    Embedder,
    FakeEmbedder,
    build_embedder,
    embedder_settings_from_env,
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
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        emb = build_embedder(mode="auto", dim=10)
        assert isinstance(emb, FakeEmbedder)
        assert emb.dim == 10

    def test_auto_with_key_is_api(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_API_KEY, "sk-test")
        emb = build_embedder(mode="auto", api_dim=DEFAULT_API_DIM)
        assert isinstance(emb, ApiEmbedder)
        assert emb.dim == DEFAULT_API_DIM
        assert emb.base_url == DEFAULT_BASE_URL

    def test_api_without_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        with pytest.raises(ValueError):
            build_embedder(mode="api")

    def test_api_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        emb = build_embedder(mode="api", api_key="sk-explicit")
        assert isinstance(emb, ApiEmbedder)

    def test_api_custom_base_url_and_dim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        emb = build_embedder(
            mode="api",
            api_key="sk-explicit",
            base_url="https://example.test/v1",
            api_dim=512,
        )
        assert isinstance(emb, ApiEmbedder)
        assert emb.base_url == "https://example.test/v1"
        assert emb.dim == 512

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            build_embedder(mode="nonsense")


class TestSettingsFromEnv:
    def test_defaults_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_BASE_URL, raising=False)
        monkeypatch.delenv(ENV_MODEL, raising=False)
        monkeypatch.delenv(ENV_DIM, raising=False)
        settings = embedder_settings_from_env()
        assert settings["base_url"] == DEFAULT_BASE_URL
        assert settings["model"] == "voyageai/voyage-4-lite"
        assert settings["api_dim"] == DEFAULT_API_DIM

    def test_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_BASE_URL, "https://custom/v1")
        monkeypatch.setenv(ENV_MODEL, "intfloat/e5-large-v2")
        monkeypatch.setenv(ENV_DIM, "1024")
        settings = embedder_settings_from_env()
        assert settings["base_url"] == "https://custom/v1"
        assert settings["model"] == "intfloat/e5-large-v2"
        assert settings["api_dim"] == 1024

    def test_invalid_dim_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_DIM, "not-a-number")
        assert embedder_settings_from_env()["api_dim"] == DEFAULT_API_DIM


class TestApiEmbedderContract:
    def test_no_key_embed_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Без ключа embed() → EmbedderError (ленивый клиент)."""
        from src.memory.embedder import EmbedderError

        monkeypatch.delenv(ENV_API_KEY, raising=False)
        emb = ApiEmbedder()
        with pytest.raises(EmbedderError):
            emb.embed("hello")

    def test_rejects_invalid_dim(self) -> None:
        with pytest.raises(ValueError):
            ApiEmbedder(dim=0)

    def test_normalizes_and_caches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """L2-нормировка + кэш; без сети (инъекция фейкового клиента)."""

        class _Resp:
            def __init__(self, vec: list[float]) -> None:
                self.data = [type("D", (), {"embedding": vec})()]

        calls = {"n": 0}

        class _Client:
            class embeddings:  # имитация SDK
                @staticmethod
                def create(**kwargs: object) -> _Resp:
                    calls["n"] += 1
                    return _Resp([3.0, 4.0])

        emb = ApiEmbedder(dim=2)
        emb._client = _Client()
        v = emb.embed("hello")
        assert np.linalg.norm(v) == pytest.approx(1.0)
        assert v.tolist() == pytest.approx([0.6, 0.8])
        # Повторный вызов — из кэша, без нового запроса
        emb.embed("hello")
        assert calls["n"] == 1
