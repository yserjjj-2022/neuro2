"""ConversationHistory — ring buffer of dialogue turns (Shell, in-memory).

Keeps the last ``max_turns`` messages so the prompt context does not grow
without bound. Persistence of dialogue is out of scope for S3.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable


class ConversationHistory:
    """Кольцевой буфер реплик диалога (in-memory).

    Attributes:
        max_turns: Максимум хранимых сообщений (роль-сообщение = 1 запись).
    """

    def __init__(self, max_turns: int = 20) -> None:
        if max_turns < 0:
            raise ValueError(f"max_turns must be >= 0, got {max_turns}")
        self.max_turns = max_turns
        self._messages: deque[dict] = deque(maxlen=max_turns if max_turns else None)

    def add_user(self, text: str) -> None:
        """Добавить реплику собеседника.

        Args:
            text: Текст сообщения.
        """
        if self.max_turns:
            self._messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        """Добавить реплику хоста.

        Args:
            text: Текст ответа.
        """
        if self.max_turns:
            self._messages.append({"role": "assistant", "content": text})

    def clear(self) -> None:
        """Очистить историю."""
        self._messages.clear()

    def as_messages(self) -> list[dict]:
        """Снимок истории для chat-API.

        Returns:
            Копия списка [{role, content}, ...].
        """
        return [dict(m) for m in self._messages]

    def __len__(self) -> int:
        """Число сообщений в буфере."""
        return len(self._messages)

    def extend(self, messages: Iterable[dict]) -> None:
        """Добавить несколько сообщений (для тестов/восстановления).

        Args:
            messages: Итерируемое [{role, content}, ...].
        """
        for m in messages:
            self._messages.append(dict(m))
