"""MemoryRouter — imperative shell orchestrating episodic memory.

Wires ``MemoryStore`` + ``Embedder`` into the host tick:
    recall → prior (bounded vector appended to the bus)
    significant event → store episode

Design: memory must never crash the tick. Unlike ``MemoryStore`` (fail-fast for
its caller), the router catches ``MemoryStoreError`` / ``EmbedderError``, logs
and returns a safe default — a memory hiccup should not stop the organism.
"""

from __future__ import annotations

import logging

import numpy as np

from .embedder import Embedder, EmbedderError
from .errors import MemoryStoreError
from .events import build_event_content, is_significant_event
from .models import Episode
from .prior import MEMORY_PRIOR_DIM, encode_memory_prior
from .protocols import SupportsConsolidate, SupportsRecall, SupportsStore
from .serialize import Vector

logger = logging.getLogger(__name__)


class MemoryRouter:
    """Оркестрация памяти: recall → приор, запись на значимых событиях.

    Attributes:
        store: Хранилище эпизодов (store + recall).
        embedder: Преобразователь текста в вектор.
        spike_threshold: Порог всплеска F для значимости события.
        recall_limit: Сколько эпизодов извлекать при recall.
        prior_dim: Размерность приора (по умолчанию ``MEMORY_PRIOR_DIM``).
    """

    def __init__(
        self,
        store: SupportsStore & SupportsRecall & SupportsConsolidate,
        embedder: Embedder,
        spike_threshold: float,
        recall_limit: int = 1,
        prior_dim: int = MEMORY_PRIOR_DIM,
    ) -> None:
        if spike_threshold < 0.0:
            raise ValueError(f"spike_threshold must be >= 0, got {spike_threshold}")
        if recall_limit < 1:
            raise ValueError(f"recall_limit must be >= 1, got {recall_limit}")
        if prior_dim <= 0:
            raise ValueError(f"prior_dim must be > 0, got {prior_dim}")
        self.store = store
        self.embedder = embedder
        self.spike_threshold = spike_threshold
        self.recall_limit = recall_limit
        self.prior_dim = prior_dim
        self._cached_text: str | None = None
        self._cached_embedding: Vector | None = None

    def context_embedding(self, text: str) -> Vector | None:
        """Эмбеддинг текста с кэшем (повторный текст не эмбеддится).

        Args:
            text: Текущий текст собеседника ("" → нет контекста).

        Returns:
            Вектор эмбеддинга или None, если текста нет / сбой эмбеддера.
        """
        if not text:
            return None
        if text == self._cached_text:
            return self._cached_embedding
        try:
            embedding = self.embedder.embed(text)
        except EmbedderError as exc:
            logger.error("memory: embed failed (%s)", exc)
            return None
        self._cached_text = text
        self._cached_embedding = embedding
        return embedding

    def recall_prior(self, query: Vector | None) -> Vector:
        """Приор из наиболее похожего прошлого эпизода.

        Args:
            query: Вектор запроса (None / нулевой → нулевой приор).

        Returns:
            Вектор shape=(prior_dim,) ∈ [-1, 1]; нули, если памяти нет
            или recall ничего не нашёл / упал.
        """
        if query is None or float(np.linalg.norm(query)) == 0.0:
            return np.zeros(self.prior_dim, dtype=np.float64)
        try:
            episodes = self.store.recall(query, limit=self.recall_limit)
        except MemoryStoreError as exc:
            logger.error("memory: recall failed (%s)", exc)
            return np.zeros(self.prior_dim, dtype=np.float64)

        if not episodes:
            return np.zeros(self.prior_dim, dtype=np.float64)
        try:
            return encode_memory_prior(episodes[0], query)
        except ValueError as exc:
            logger.error("memory: prior encode failed (%s)", exc)
            return np.zeros(self.prior_dim, dtype=np.float64)

    def episode_count(self) -> int:
        """Число эпизодов в хранилище (для ночного цикла, S6 проход 2).

        Returns:
            Число эпизодов; 0 при сбое (память не роняет тик).
        """
        try:
            return self.store.count()
        except (MemoryStoreError, AttributeError) as exc:
            logger.error("memory: count failed (%s)", exc)
            return 0

    def maybe_store(
        self,
        *,
        text: str,
        query: Vector | None,
        f: float,
        prev_f: float,
        valence: float,
        stress: float,
        active_tags: tuple[str, ...],
        reflex_tags: tuple[str, ...],
        now: float,
        has_new_message: bool = False,
    ) -> int | None:
        """Записать эпизод, если событие значимо.

        ``content`` = текст собеседника (если есть), иначе дескриптор события.
        ``embedding`` = query (если есть), иначе эмбеддинг content.

        Args:
            text: Текущий текст собеседника.
            query: Эмбеддинг текста (если был).
            f: Текущее F(t).
            prev_f: F(t-1).
            valence: Валентность.
            stress: Стресс.
            active_tags: Активные сегменты шины.
            reflex_tags: Критические сигналы.
            now: Текущее время (synthetic: tick·dt; wall: clock) → timestamp.
            has_new_message: Пришло ли новое сообщение (значимо всегда).

        Returns:
            id записанного эпизода или None (незначимо / сбой).

        Note:
            Ошибки store логируются и не пробрасываются: память не должна
            ронять тик (в отличие от прямого ``MemoryStore.store``).
        """
        if not is_significant_event(
            f, prev_f, reflex_tags, self.spike_threshold, has_new_message
        ):
            return None

        content = text or build_event_content(active_tags, reflex_tags, valence, stress)
        embedding = query
        if embedding is None:
            try:
                embedding = self.embedder.embed(content)
            except EmbedderError as exc:
                logger.error("memory: embed for store failed (%s)", exc)
                return None

        episode = Episode(
            content=content,
            embedding=embedding,
            timestamp=float(now),
            valence=valence,
            stress=stress,
            free_energy=f,
        )
        try:
            return self.store.store(episode)
        except (MemoryStoreError, ValueError) as exc:
            logger.error("memory: store failed (%s)", exc)
            return None
