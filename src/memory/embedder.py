"""Embedder — text → vector, the keystone that unblocks memory wiring.

Functional Core / Imperative Shell (ADR-0004):
    - ``Embedder`` — Protocol for DI (like ``SupportsWrite`` in telemetry).
    - ``FakeEmbedder`` — deterministic core: bag-of-tokens hashing. Same text →
      same vector; similar texts (shared tokens) → higher cosine. Used in tests
      and replay, no network, no keys.
    - ``ApiEmbedder`` — shell: OpenAI-compatible client (RouterAI by default).
      Real semantic embeddings for live runs.
    - ``build_embedder`` — factory honouring ``embedder_mode`` (``auto``/``fake``/
      ``api``). ``auto`` picks API when a key is configured, else fake, so tests
      stay deterministic while live runs get a real embedder.

Configuration is read from the environment (``.env`` is loaded by the CLI):
    EMBEDDER_API_KEY   — API key (required for api mode)
    EMBEDDER_BASE_URL  — OpenAI-compatible base URL (default: RouterAI)
    EMBEDDER_MODEL     — embedding model id (default: voyageai/voyage-4-lite)
    EMBEDDER_DIM       — output vector dimension (Matryoshka, default: 256)

``embedder_mode="api"`` without a key is fail-fast (``ValueError``); ``auto``
without a key silently falls back to fake (not an error).
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from .serialize import Vector

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

# Дефолты RouterAI (OpenAI-совместимый шлюз). Переопределяются через .env.
DEFAULT_BASE_URL = "https://routerai.ru/api/v1"
DEFAULT_API_MODEL = "voyageai/voyage-4-lite"
DEFAULT_API_DIM = 256

# Имена переменных окружения (единая точка правды).
ENV_API_KEY = "EMBEDDER_API_KEY"
ENV_BASE_URL = "EMBEDDER_BASE_URL"
ENV_MODEL = "EMBEDDER_MODEL"
ENV_DIM = "EMBEDDER_DIM"


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
    """OpenAI-совместимый эмбеддер (Shell, event-triggered, lazy client).

    По умолчанию — RouterAI (https://routerai.ru). Клиент создаётся при
    первом вызове ``embed`` (не в ``__init__``), чтобы создание без ключа не
    падало до реального использования.

    Вектор L2-нормируется (как ``FakeEmbedder``): масштаб входа не зависит
    от модели и размерности, а косинус равен скалярному произведению. Это
    держит F(t) в предсказуемом диапазоне и делает пороги переносимыми
    между эмбеддерами.

    Тестами НЕ покрывается (сеть): проверяется контракт и фабрика.

    Attributes:
        model: Идентификатор модели эмбеддингов (напр. ``voyageai/voyage-4-lite``).
        dim: Ожидаемая размерность ответа (усечение Matryoshka через
            ``dimensions``, если модель поддерживает).
        base_url: OpenAI-совместимый base URL.
        api_key: Ключ API; None → из ``EMBEDDER_API_KEY``.
        normalize: L2-нормировать вектор (по умолчанию True).
    """

    model: str = DEFAULT_API_MODEL
    dim: int = DEFAULT_API_DIM
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    normalize: bool = True
    _client: OpenAI | None = field(default=None, init=False, repr=False)
    _cache: dict[str, Vector] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim must be > 0, got {self.dim}")

    def _ensure_client(self) -> OpenAI:
        """Лениво создать клиент (первый вызов embed).

        Returns:
            Клиент OpenAI.

        Raises:
            EmbedderError: Если ключ не найден или пакет openai недоступен.
        """
        if self._client is not None:
            return self._client

        key = self.api_key or os.environ.get(ENV_API_KEY)
        if not key:
            raise EmbedderError(f"no API key: pass api_key or set {ENV_API_KEY}")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - зависимость объявлена
            raise EmbedderError(f"openai package unavailable: {exc}") from exc

        self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    def embed(self, text: str) -> Vector:
        """Эмбеддинг текста через OpenAI-совместимый API (с кэшем).

        Активное сообщение не меняется между тиками, а вызов API дорог —
        одинаковый текст эмбеддится один раз.

        Args:
            text: Входной текст.

        Returns:
            Вектор shape=(dim,) float64.

        Raises:
            EmbedderError: При сбое сети/API или неверной размерности ответа.
        """
        if text in self._cache:
            return self._cache[text]

        client = self._ensure_client()
        try:
            response = client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dim,
                encoding_format="float",
            )
        except Exception as exc:  # любые сбои → EmbedderError
            raise EmbedderError(f"embedding request failed: {exc}") from exc

        data = np.asarray(response.data[0].embedding, dtype=np.float64)
        if data.shape[0] != self.dim:
            raise EmbedderError(f"embedding dim {data.shape[0]} != expected {self.dim}")
        if self.normalize:
            norm = float(np.linalg.norm(data))
            if norm > 0.0:
                data = data / norm
        self._cache[text] = data
        return data


def build_embedder(
    mode: str = "auto",
    dim: int = 8,
    model: str = DEFAULT_API_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_dim: int = DEFAULT_API_DIM,
    api_key: str | None = None,
) -> Embedder:
    """Фабрика эмбеддера по режиму.

    Args:
        mode: "auto" (ключ→api, иначе fake), "fake", "api".
        dim: Размерность fake-эмбеддера.
        model: Модель API.
        base_url: OpenAI-совместимый base URL.
        api_dim: Размерность API-эмбеддинга (Matryoshka).
        api_key: Ключ API; None → из окружения (``EMBEDDER_API_KEY``).

    Returns:
        Embedder согласно режиму.

    Raises:
        ValueError: Если mode неизвестен, или mode="api" без ключа.
    """
    key = api_key or os.environ.get(ENV_API_KEY)
    if mode == "fake":
        return FakeEmbedder(dim=dim)
    if mode == "api":
        if not key:
            raise ValueError(
                f"embedder_mode='api' requires a key (argument or {ENV_API_KEY})"
            )
        return ApiEmbedder(model=model, dim=api_dim, base_url=base_url, api_key=key)
    if mode == "auto":
        if key:
            logger.info(
                "embedder: auto → api (model=%s, base_url=%s, dim=%d)",
                model,
                base_url,
                api_dim,
            )
            return ApiEmbedder(model=model, dim=api_dim, base_url=base_url, api_key=key)
        logger.info("embedder: auto → fake (dim=%d, no API key)", dim)
        return FakeEmbedder(dim=dim)
    raise ValueError(f"unknown embedder_mode {mode!r} (auto|fake|api)")


def embedder_settings_from_env() -> dict[str, object]:
    """Собрать настройки API-эмбеддера из окружения (для wiring/CLI).

    Returns:
        Словарь с ``base_url``, ``model``, ``api_dim`` (из env, с дефолтами).
        ``api_dim`` парсится как int; некорректное значение → дефолт.
    """
    dim_raw = os.environ.get(ENV_DIM)
    try:
        api_dim = int(dim_raw) if dim_raw else DEFAULT_API_DIM
    except ValueError:
        logger.warning(
            "invalid %s=%r — using default %d", ENV_DIM, dim_raw, DEFAULT_API_DIM
        )
        api_dim = DEFAULT_API_DIM
    return {
        "base_url": os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL),
        "model": os.environ.get(ENV_MODEL, DEFAULT_API_MODEL),
        "api_dim": api_dim,
    }
