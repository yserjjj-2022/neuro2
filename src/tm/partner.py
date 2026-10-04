"""PartnerModel — imperative shell owning accumulated partner signatures (S5).

Wires ``Embedder`` + pure ``compute`` into the dialogue path. Identity is a
derived regularity (ADR-0008): the model matches each utterance against its
signatures and updates the matched one; a new signature is created when nothing
matches. It never crashes the chat — embedder failures degrade to a safe
``PartnerState``.
"""

from __future__ import annotations

import logging

from src.memory.embedder import Embedder, EmbedderError

from .compute import match_partner, update_signature, update_trust
from .models import PartnerSignature, PartnerState

logger = logging.getLogger(__name__)


class PartnerModel:
    """Модель партнёра: сигнатуры + состояние (Shell, S5).

    Attributes:
        embedder: Преобразователь текста в вектор.
        match_threshold: Порог косинусной близости для узнавания.
        learning_rate: Скорость обновления сигнатуры.
        trust_gain: Прирост доверия при согласии.
        trust_decay: Утечка доверия.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        match_threshold: float = 0.75,
        learning_rate: float = 0.2,
        trust_gain: float = 0.1,
        trust_decay: float = 0.01,
    ) -> None:
        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError(
                f"match_threshold must be in [0, 1], got {match_threshold}"
            )
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(
                f"learning_rate must be in (0, 1], got {learning_rate}"
            )
        if trust_gain < 0.0 or trust_decay < 0.0:
            raise ValueError(
                f"trust_gain/trust_decay must be >= 0, got {trust_gain}/{trust_decay}"
            )
        self.embedder = embedder
        self.match_threshold = match_threshold
        self.learning_rate = learning_rate
        self.trust_gain = trust_gain
        self.trust_decay = trust_decay
        self._signatures: list[PartnerSignature] = []
        self._current: int | None = None
        self._state = PartnerState()

    @property
    def signatures(self) -> tuple[PartnerSignature, ...]:
        """Накопленные сигнатуры (только чтение)."""
        return tuple(self._signatures)

    @property
    def state(self) -> PartnerState:
        """Последнее состояние партнёра."""
        return self._state

    def observe(
        self,
        text: str,
        *,
        pause_s: float = 0.0,
        valence: float = 0.0,
        stress: float = 0.0,
    ) -> PartnerState:
        """Обработать реплику партнёра: узнать/обновить сигнатуру.

        Args:
            text: Текст реплики партнёра ("" → состояние не меняется).
            pause_s: Интервал с прошлой реплики, с.
            valence: Реакция хоста (валентность).
            stress: Стресс в момент реплики.

        Returns:
            Обновлённое ``PartnerState``. При сбое эмбеддера — безопасный
            дефолт (``uncertainty=1.0``), чат не роняется.
        """
        if not text:
            return self._state
        try:
            embedding = self.embedder.embed(text)
        except EmbedderError as exc:
            logger.error("tm: embed failed (%s)", exc)
            self._state = PartnerState(
                trust=self._state.trust,
                ambiguity=self._state.ambiguity,
                conflict=self._state.conflict,
                uncertainty=1.0,
                name=self._state.name,
            )
            return self._state

        index, similarity = match_partner(
            embedding, self._signatures, threshold=self.match_threshold
        )
        if index is None:
            self._signatures.append(
                update_signature(
                    None,
                    embedding,
                    pause_s=pause_s,
                    valence=valence,
                    stress=stress,
                    learning_rate=self.learning_rate,
                )
            )
            self._current = len(self._signatures) - 1
            uncertainty = min(1.0, max(0.0, 1.0 - max(similarity, 0.0)))
        else:
            self._signatures[index] = update_signature(
                self._signatures[index],
                embedding,
                pause_s=pause_s,
                valence=valence,
                stress=stress,
                learning_rate=self.learning_rate,
            )
            self._current = index
            uncertainty = min(1.0, max(0.0, 1.0 - max(similarity, 0.0)))

        # Конфликт = неузнавание; двусмысленность = близость у границы порога.
        conflict = 1.0 if index is None else 0.0
        ambiguity = 0.0
        if index is not None and similarity < self.match_threshold + 0.1:
            ambiguity = (self.match_threshold + 0.1 - similarity) / 0.1

        self._state = update_trust(
            self._state,
            conflict=conflict,
            ambiguity=min(1.0, max(0.0, ambiguity)),
            trust_gain=self.trust_gain,
            trust_decay=self.trust_decay,
        )
        self._state = PartnerState(
            trust=self._state.trust,
            ambiguity=self._state.ambiguity,
            conflict=self._state.conflict,
            uncertainty=uncertainty,
            name=self._state.name,
        )
        return self._state

    def set_name(self, name: str, aliases: tuple[str, ...] = ()) -> None:
        """Привязать объявленное имя/алиасы к текущей сигнатуре (ADR-0008 §4).

        Имя — объявленный символ-якорь, не выводимый; привязывается к
        сигнатуре, к которой матчится партнёр. Без сигнатуры — только в
        состояние.

        Args:
            name: Объявленное имя ("" → сброс).
            aliases: Дополнительные алиасы.
        """
        self._state = PartnerState(
            trust=self._state.trust,
            ambiguity=self._state.ambiguity,
            conflict=self._state.conflict,
            uncertainty=self._state.uncertainty,
            name=name,
        )
        if self._current is None or not name:
            return
        signature = self._signatures[self._current]
        self._signatures[self._current] = PartnerSignature(
            centroid=signature.centroid,
            weight=signature.weight,
            mean_pause_s=signature.mean_pause_s,
            mean_valence=signature.mean_valence,
            mean_stress=signature.mean_stress,
            name=name,
            aliases=aliases,
        )
