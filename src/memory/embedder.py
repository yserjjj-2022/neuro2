"""Embedder — text → vector, the keystone that unblocks memory wiring.

Functional Core / Imperative Shell (ADR-0004):
    - ``Embedder`` — Protocol for DI (like ``SupportsWrite`` in telemetry).
    - ``FakeEmbedder`` — deterministic core: bag-of-tokens hashing. Same text →
      same vector; similar texts (shared tokens) → higher cosine. Used in tests
      and replay, no network, no keys.
    - ``ApiEmbedder`` — shell: OpenAI embeddings, lazy client. Real semantic
      embeddings for live runs.
    - ``build_embedder`` — factory honouring ``embedder_mode`` (``auto``/``fake``/
      ``api``). ``auto`` picks API when a key is available, else fake, so tests
      stay deterministic while live runs get a real embedder.

``embedder_mode="api"`` without a key is fail-fast (``ValueError``); ``auto``
without a key silently falls back to fake (not an error).
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .serialize import Vector

logger = logging.getLogger(__name__)

# Модель и размерность API по умолчанию (OpenAI text-embedding-3-small).
DEFAULT_API_MODEL = "text-embedding-3-small"
DEFAULT_API_DIM = 1536
_API_KEY_ENV = "OPENAI_API_KEY"


class EmbedderError(Exception):
    """Оборачивает сбои эмбеддера (сеть, API, размерность ответа)."""


@runtime_checkable
class Embedder(Protocol):
    """Контракт эмбеддера: текст → вектор.

    Attributes:
        dim: Размерность возвращаемых векторов.
    """

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> Vector:
        """Эмбеддинг текста.

        Args:
            text: Входной текст ("" допустим → вектор нулей).

        Returns:
            Вектор shape=(dim,), конечные значения.

        Raises:
            EmbedderError: При сбое эмбеддера.
        """
        ...


@dataclass(frozen=True)
class FakeEmbedder:
    """Детерминированный эмбеддер для тестов и replay (Core).

    Bag-of-tokens hashing: каждый токен хешируется (sha256) в бакет
    ``[0, dim)`` со знаком от ещё одного бита хеша; вектор L2-нормируется.
    Общие токены → общий вклад → похожие тексты дают больший косинус.

    Пустой текст (или текст без токенов) → вектор нулей (не ошибка).

    Attributes:
        dim: Размерность вектора.
        seed: Зерно (меняет раскладку бакетов, сохраняя детерминизм).
    """

    dim: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be > 0, got {self.dim}")

    def embed(self, text: str) -> Vector:
        """Детерминированный эмбеддинг через bag-of-tokens hashing.

        Args:
            text: Входной текст.

        Returns:
            Вектор shape=(dim,) float64; нули для пустого текста.
        """
        vec = np.zeros(self.dim, dtype=np.float64)
        tokens = text.lower().split()
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.sha256(f"{self.seed}:{token}".encode()).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec


@dataclass
class ApiEmbedder:
    """OpenAI embeddings (Shell, event-triggered, lazy client).

    Клиент создаётся при первом вызове ``embed`` (не в ``__init__``), чтобы
    импорт/создание без ключа не падало до реального использования. Ключ
    берётся из аргумента или ``OPENAI_API_KEY``.

    Тестами НЕ покрывается (сеть): проверяется контракт и фабрика.

    Attributes:
        model: Имя модели эмбеддингов.
        dim: Ожидаемая размерность ответа.
        api_key: Ключ API; None → из окружения.
    """

    model: str = DEFAULT_API_MODEL
    dim: int = DEFAULT_API_DIM
    api_key: str | None = None
    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be > 0, got {self.dim}")

    def _ensure_client(self) -> object:
        """Лениво создать клиент OpenAI (первый вызов embed).

        Returns:
            Клиент OpenAI.

        Raises:
            EmbedderError: Если ключ не найден или пакет openai недоступен.
        """
        if self._client is not None:
            return self._client

        key = self.api_key or os.environ.get(_API_KEY_ENV)
        if not key:
            raise EmbedderError(f"no API key: pass api_key or set {_API_KEY_ENV}")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - зависимость объявлена
            raise EmbedderError(f"openai package unavailable: {exc}") from exc

        self._client = OpenAI(api_key=key)
        return self._client

    def embed(self, text: str) -> Vector:
        """Эмбеддинг текста через OpenAI API.

        Args:
            text: Входной текст.

        Returns:
            Вектор shape=(dim,) float64.

        Raises:
            EmbedderError: При сбое сети/API или неверной размерности ответа.
        """
        client = self._ensure_client()
        try:
            response = client.embeddings.create(model=self.model, input=text)
        except Exception as exc:
            raise EmbedderError(f"embedding request failed: {exc}") from exc

        data = np.asarray(response.data[0].embedding, dtype=np.float64)
        if data.shape[0] != self.dim:
            raise EmbedderError(f"embedding dim {data.shape[0]} != expected {self.dim}")
        return data


def build_embedder(
    mode: str = "auto",
    dim: int = 8,
    model: str = DEFAULT_API_MODEL,
    api_key: str | None = None,
) -> Embedder:
    """Фабрика эмбеддера по режиму.

    Args:
        mode: "auto" (ключ→api, иначе fake), "fake", "api".
        dim: Размерность fake-эмбеддера.
        model: Модель API.
        api_key: Ключ API; None → из окружения.

    Returns:
        Embedder согласно режиму.

    Raises:
        ValueError: Если mode неизвестен, или mode="api" без ключа.
    """
    if mode == "fake":
        return FakeEmbedder(dim=dim)
    if mode == "api":
        if not (api_key or os.environ.get(_API_KEY_ENV)):
            raise ValueError(
                f"embedder_mode='api' requires a key (argument or {_API_KEY_ENV})"
            )
        return ApiEmbedder(model=model, dim=DEFAULT_API_DIM, api_key=api_key)
    if mode == "auto":
        if api_key or os.environ.get(_API_KEY_ENV):
            logger.info("embedder: auto → api (model=%s)", model)
            return ApiEmbedder(model=model, dim=DEFAULT_API_DIM, api_key=api_key)
        logger.info("embedder: auto → fake (dim=%d, no API key)", dim)
        return FakeEmbedder(dim=dim)
    raise ValueError(f"unknown embedder_mode {mode!r} (auto|fake|api)")
