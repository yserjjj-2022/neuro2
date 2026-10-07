"""SpeechController — event-triggered speech generation (Shell).

Decides whether to speak, recalls relevant precedents, builds an IntentFrame,
renders chat messages and calls the LLM. Speech must never crash the loop:
LLM/recall failures are logged and yield ``None``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from src.memory.embedder import Embedder, EmbedderError
from src.memory.errors import MemoryStoreError
from src.memory.protocols import SupportsRecall
from src.speech.intent import (
    build_intent_frame,
    register_max_tokens,
    render_messages,
)
from src.speech.llm import LlmClient, LlmError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechDecision:
    """Решение о реплике.

    Attributes:
        speak: Говорить ли.
        reason: Причина ("new_message", "initiative", "silent").
    """

    speak: bool
    reason: str


def should_speak(
    *,
    new_message: bool,
    f: float,
    f_threshold: float,
) -> SpeechDecision:
    """Решить, говорить ли на этом событии.

    Новое сообщение → всегда отвечаем. Иначе — инициатива, если F выше
    порога (заготовка; полноценная policy — S4).

    Args:
        new_message: Пришло ли новое сообщение оператора.
        f: Текущее F(t).
        f_threshold: Порог F для инициативы.

    Returns:
        SpeechDecision.
    """
    if new_message:
        return SpeechDecision(speak=True, reason="new_message")
    if f > f_threshold:
        return SpeechDecision(speak=True, reason="initiative")
    return SpeechDecision(speak=False, reason="silent")


class SpeechController:
    """Оркестрация речи: recall прецедентов → frame → LLM.

    Attributes:
        llm: Клиент генерации.
        memory: Источник прецедентов (recall); None → без прецедентов.
        embedder: Преобразователь текста в вектор для recall; None → без recall.
        f_threshold: Порог F для инициативы.
        recall_limit: Сколько прецедентов извлекать.
        default_register: Речевой режим по умолчанию.
    """

    def __init__(
        self,
        llm: LlmClient,
        memory: SupportsRecall | None = None,
        embedder: Embedder | None = None,
        f_threshold: float = 1.0,
        recall_limit: int = 3,
        default_register: str = "brief",
    ) -> None:
        if f_threshold < 0.0:
            raise ValueError(f"f_threshold must be >= 0, got {f_threshold}")
        if recall_limit < 1:
            raise ValueError(f"recall_limit must be >= 1, got {recall_limit}")
        register_max_tokens(default_register)  # валидация режима
        self.llm = llm
        self.memory = memory
        self.embedder = embedder
        self.f_threshold = f_threshold
        self.recall_limit = recall_limit
        self.default_register = default_register

    def recall_precedents(self, user_text: str) -> tuple[str, ...]:
        """Content релевантных прошлых эпизодов.

        Args:
            user_text: Текущее сообщение (запрос recall).

        Returns:
            Кортеж content прецедентов; пусто при отсутствии памяти/сбое.
        """
        if self.memory is None or self.embedder is None or not user_text:
            return ()
        try:
            query = self.embedder.embed(user_text)
            episodes = self.memory.recall(query, limit=self.recall_limit)
        except (EmbedderError, MemoryStoreError) as exc:
            logger.error("speech: recall failed (%s)", exc)
            return ()
        return tuple(ep.content for ep in episodes)

    def respond(
        self,
        *,
        user_text: str,
        f: float,
        valence: float,
        stress: float,
        task: str,
        history: Sequence[dict] = (),
        register: str | None = None,
        new_message: bool = True,
        goal: str | None = None,
        partner_name: str = "",
        decision: SpeechDecision | None = None,
    ) -> str | None:
        """Событийный ответ: frame → messages → LLM.

        Args:
            user_text: Текущее сообщение собеседника.
            f: Свободная энергия F(t).
            valence: Валентность.
            stress: Аллостатический стресс.
            task: Активная задача/аттрактор.
            history: Предыдущие сообщения диалога.
            register: Речевой режим (None → default_register).
            new_message: Пришло ли новое сообщение.
            goal: Цель реплики из policy (S4); None → "respond" (S3).
            partner_name: Принятое имя партнёра (S5); "" → без вокатива.
            decision: Готовое решение о речи (policy — S4). Если задано,
                ``should_speak`` **не** вызывается: policy — единственный
                авторитет, ``should_speak`` — рудимент S3 (fallback при
                ``decision=None``).

        Returns:
            Текст ответа или None (не отвечаем / сбой LLM).
        """
        if decision is None:
            # S3-fallback: policy не решала (policy=None/disabled в ChatSession).
            decision = should_speak(
                new_message=new_message, f=f, f_threshold=self.f_threshold
            )
        if not decision.speak:
            return None

        register = register or self.default_register
        precedents = self.recall_precedents(user_text)
        frame = build_intent_frame(
            f=f,
            valence=valence,
            stress=stress,
            task=task,
            precedents=precedents,
            register=register,
            goal=goal or "respond",
            partner_name=partner_name,
        )
        messages = render_messages(frame, user_text, history=history)
        try:
            return self.llm.reply(messages, max_tokens=register_max_tokens(register))
        except LlmError as exc:
            logger.error("speech: LLM failed (%s)", exc)
            return None
