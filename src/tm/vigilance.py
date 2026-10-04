"""VigilanceGate — new claims as hypotheses until confirmed (S5, §4).

Imperative shell over the pure ``detect_conflict``: it keeps the accumulated
embeddings of what the host "knows" and, for each new claim, computes the
conflict and marks it a hypothesis. It never blocks a reply and never accuses
(manifest §3.Г): behaviour is driven by the raw signal, the label is
communication (ADR-0008 §1). The LLM does not reason here (ADR-0007).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from src.memory.embedder import Embedder, EmbedderError

from .compute import detect_conflict
from .models import Claim

logger = logging.getLogger(__name__)


class VigilanceGate:
    """Ворота бдительности: утверждения — гипотезы до подтверждения.

    Attributes:
        embedder: Преобразователь текста в вектор.
        conflict_threshold: Порог рассогласования для «спорной» гипотезы.
        max_memory: Сколько последних утверждений держать в накопленном.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        conflict_threshold: float = 0.6,
        max_memory: int = 64,
    ) -> None:
        if not 0.0 <= conflict_threshold <= 1.0:
            raise ValueError(
                f"conflict_threshold must be in [0, 1], got {conflict_threshold}"
            )
        if max_memory < 1:
            raise ValueError(f"max_memory must be >= 1, got {max_memory}")
        self.embedder = embedder
        self.conflict_threshold = conflict_threshold
        self.max_memory = max_memory
        self._memory: list[np.ndarray] = []
        self._last_claim: Claim | None = None

    @property
    def last_claim(self) -> Claim | None:
        """Последнее обработанное утверждение (гипотеза)."""
        return self._last_claim

    def observe(self, text: str) -> Claim:
        """Оценить утверждение как гипотезу.

        Высокий конфликт с накопленным → низкая ``confidence`` (гипотеза под
        сомнением); согласное утверждение → высокая ``confidence``. Само
        утверждение добавляется в накопленное (память растёт).

        Args:
            text: Текст утверждения ("" → пустая гипотеза без изменений).

        Returns:
            Claim. При сбое эмбеддера — безопасная гипотеза (confidence=0.0,
            conflict=1.0), контур не роняется.
        """
        if not text:
            return self._last_claim or Claim(content="", confidence=0.0, conflict=1.0)
        try:
            embedding = np.asarray(self.embedder.embed(text), dtype=np.float64)
        except EmbedderError as exc:
            logger.error("vigilance: embed failed (%s)", exc)
            self._last_claim = Claim(content=text, confidence=0.0, conflict=1.0)
            return self._last_claim

        conflict = detect_conflict(embedding, self._memory)
        confidence = float(min(1.0, max(0.0, 1.0 - conflict)))
        claim = Claim(content=text, confidence=confidence, conflict=conflict)

        self._memory.append(embedding)
        if len(self._memory) > self.max_memory:
            self._memory.pop(0)
        self._last_claim = claim
        return claim

    def confirm(self) -> Claim | None:
        """Подтвердить последнюю гипотезу практикой (повышает confidence).

        Returns:
            Обновлённый Claim или None, если гипотез не было.
        """
        if self._last_claim is None:
            return None
        self._last_claim = Claim(
            content=self._last_claim.content,
            confidence=1.0,
            conflict=self._last_claim.conflict,
            confirmed=True,
        )
        return self._last_claim

    def is_hypothesis(self) -> bool:
        """Является ли последнее утверждение неподтверждённой гипотезой.

        Истина, если конфликт выше порога и утверждение ещё не подтверждено.
        """
        claim = self._last_claim
        if claim is None:
            return False
        return not claim.confirmed and claim.conflict >= self.conflict_threshold

    def memory(self) -> Sequence[np.ndarray]:
        """Накопленные эмбеддинги (read-only копия)."""
        return tuple(self._memory)
