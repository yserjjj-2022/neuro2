"""TextMessageProvider — communicative input as real text → embedding.

Replaces the placeholder ``UserMessageProvider`` (fixed random vector) with a
real embedder call, unblocking both the communicative route and memory wiring
(``[Phase1][memory-wiring]``). The scripted ``messages`` keep the provider a
deterministic function of ``(tick, now)`` for tests/replay; in S3 a live input
swaps in without changing the interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.mcp import SignalCategory, SignalSource
from src.memory.embedder import Embedder, EmbedderError


@dataclass(frozen=True)
class TextMessageProvider:
    """Сообщение собеседника: текст → эмбеддинг в шину.

    Детерминирован при детерминированном эмбеддере. ``messages`` — скрипт
    ``(tick, text)`` для mock/тестов; сообщение остаётся активным до
    следующего (последнее с ``tick' <= tick``).

    Повторные вызовы для одного текста дёшевы: ``ApiEmbedder`` кэширует
    результат (активное сообщение не меняется между тиками).

    Attributes:
        embedder: Преобразователь текста в вектор.
        messages: Скрипт сообщений ``(tick, text)`` в произвольном порядке.
        tag: Идентификатор источника.
        category: Коммуникативный.
        period: Обновляется каждый тик.
    """

    embedder: Embedder
    messages: tuple[tuple[int, str], ...] = ()
    tag: str = "user_message"
    category: SignalCategory = SignalCategory.COMMUNICATIVE
    period: int = 1

    @property
    def dim(self) -> int:
        """Размерность: dim эмбеддера."""
        return self.embedder.dim

    def text_at(self, tick: int) -> str:
        """Текст последнего сообщения с ``tick' <= tick``.

        Args:
            tick: Номер текущего тика.

        Returns:
            Текст активного сообщения или "" (сообщений ещё не было).
        """
        active = ""
        for at, text in self.messages:
            if at <= tick:
                active = text
        return active

    def read(self, tick: int, now: float) -> SignalSource:
        """Эмбеддинг активного сообщения как сигнал шины.

        Args:
            tick: Номер тика (определяет активное сообщение).
            now: Wall-clock время (не используется).

        Returns:
            SignalSource shape=(dim,) с эмбеддингом текста; нули, если
            сообщения нет или эмбеддер дал сбой (коммуникативный вход
            не должен ронять тик).
        """
        text = self.text_at(tick)
        if not text:
            data = np.zeros(self.dim, dtype=np.float64)
        else:
            try:
                data = np.asarray(self.embedder.embed(text), dtype=np.float64)
            except EmbedderError:
                data = np.zeros(self.dim, dtype=np.float64)
        return SignalSource(category=self.category, data=data, tag=self.tag)
