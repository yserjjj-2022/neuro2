"""Unit tests for ConversationHistory — in-memory ring buffer."""

from __future__ import annotations

import pytest

from src.speech.history import ConversationHistory


class TestConversationHistory:
    def test_empty(self) -> None:
        history = ConversationHistory(max_turns=10)
        assert len(history) == 0
        assert history.as_messages() == []

    def test_add_and_roles(self) -> None:
        history = ConversationHistory(max_turns=10)
        history.add_user("привет")
        history.add_assistant("ага")
        assert history.as_messages() == [
            {"role": "user", "content": "привет"},
            {"role": "assistant", "content": "ага"},
        ]

    def test_ring_buffer_limit(self) -> None:
        history = ConversationHistory(max_turns=3)
        for i in range(5):
            history.add_user(f"m{i}")
        messages = history.as_messages()
        assert len(messages) == 3
        assert messages[-1]["content"] == "m4"
        assert messages[0]["content"] == "m2"

    def test_clear(self) -> None:
        history = ConversationHistory(max_turns=5)
        history.add_user("x")
        history.clear()
        assert len(history) == 0

    def test_zero_turns_disables_storage(self) -> None:
        history = ConversationHistory(max_turns=0)
        history.add_user("x")
        assert len(history) == 0

    def test_negative_turns_raises(self) -> None:
        with pytest.raises(ValueError):
            ConversationHistory(max_turns=-1)

    def test_as_messages_returns_copy(self) -> None:
        history = ConversationHistory(max_turns=5)
        history.add_user("x")
        snapshot = history.as_messages()
        snapshot.append({"role": "user", "content": "injected"})
        assert len(history) == 1

    def test_extend(self) -> None:
        history = ConversationHistory(max_turns=5)
        history.extend([{"role": "user", "content": "a"}])
        assert len(history) == 1