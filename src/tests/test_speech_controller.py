"""Unit tests for SpeechController — shell with fake LLM/memory."""

from __future__ import annotations

import numpy as np
import pytest

from src.memory.embedder import FakeEmbedder
from src.memory.errors import MemoryStoreError
from src.memory.models import Episode
from src.speech.controller import SpeechController, should_speak
from src.speech.llm import FakeLlmClient, LlmError


class FakeRecall:
    def __init__(self) -> None:
        self.episodes: list[Episode] = []
        self.fail = False

    def recall(self, query: np.ndarray, limit: int = 5) -> list[Episode]:
        if self.fail:
            raise MemoryStoreError("boom")
        return self.episodes[:limit]


def _episode(content: str, dim: int = 8) -> Episode:
    return Episode(
        content=content,
        embedding=np.ones(dim),
        timestamp=0.0,
        valence=0.0,
        stress=0.0,
        free_energy=0.0,
    )


class TestShouldSpeak:
    def test_new_message_speaks(self) -> None:
        d = should_speak(new_message=True, f=0.0, f_threshold=10.0)
        assert d.speak and d.reason == "new_message"

    def test_initiative_above_threshold(self) -> None:
        d = should_speak(new_message=False, f=5.0, f_threshold=1.0)
        assert d.speak and d.reason == "initiative"

    def test_silent_below_threshold(self) -> None:
        d = should_speak(new_message=False, f=0.5, f_threshold=1.0)
        assert not d.speak and d.reason == "silent"


class TestSpeechController:
    def _controller(
        self,
        memory: FakeRecall | None = None,
        llm: object | None = None,
    ) -> SpeechController:
        return SpeechController(
            llm=llm or FakeLlmClient(),  # type: ignore[arg-type]
            memory=memory,
            embedder=FakeEmbedder(dim=8) if memory is not None else None,
            f_threshold=1.0,
        )

    def test_recall_precedents(self) -> None:
        memory = FakeRecall()
        memory.episodes = [_episode("оператор просил краткость")]
        controller = self._controller(memory=memory)
        precedents = controller.recall_precedents("привет")
        assert precedents == ("оператор просил краткость",)

    def test_recall_without_memory_empty(self) -> None:
        assert self._controller().recall_precedents("привет") == ()

    def test_recall_failure_empty(self) -> None:
        memory = FakeRecall()
        memory.fail = True
        assert self._controller(memory=memory).recall_precedents("x") == ()

    def test_respond_returns_reply(self) -> None:
        controller = self._controller()
        reply = controller.respond(
            user_text="привет",
            f=0.0,
            valence=0.0,
            stress=0.0,
            task="tone",
        )
        assert reply is not None and reply != ""

    def test_respond_silent_without_message_below_threshold(self) -> None:
        controller = self._controller()
        reply = controller.respond(
            user_text="",
            f=0.0,
            valence=0.0,
            stress=0.0,
            task="tone",
            new_message=False,
        )
        assert reply is None

    def test_respond_initiative_above_threshold(self) -> None:
        controller = self._controller()
        reply = controller.respond(
            user_text="",
            f=5.0,
            valence=0.0,
            stress=0.0,
            task="tone",
            new_message=False,
        )
        assert reply is not None

    def test_llm_failure_returns_none(self) -> None:
        class BrokenLlm:
            def reply(self, messages: list[dict], max_tokens: int = 256) -> str:
                raise LlmError("down")

        controller = self._controller(llm=BrokenLlm())
        assert (
            controller.respond(
                user_text="привет", f=0.0, valence=0.0, stress=0.0, task="tone"
            )
            is None
        )

    def test_history_passed_to_llm(self) -> None:
        captured: dict[str, list[dict]] = {}

        class CaptureLlm:
            def reply(self, messages: list[dict], max_tokens: int = 256) -> str:
                captured["messages"] = messages
                return "ok"

        controller = self._controller(llm=CaptureLlm())
        history = [{"role": "user", "content": "старое"}]
        controller.respond(
            user_text="новое",
            f=0.0,
            valence=0.0,
            stress=0.0,
            task="tone",
            history=history,
        )
        roles = [m["role"] for m in captured["messages"]]
        assert roles == ["system", "user", "user"]

    def test_register_passed(self) -> None:
        captured: dict[str, int] = {}

        class CaptureLlm:
            def reply(self, messages: list[dict], max_tokens: int = 256) -> str:
                captured["max_tokens"] = max_tokens
                return "ok"

        controller = self._controller(llm=CaptureLlm())
        controller.respond(
            user_text="x",
            f=0.0,
            valence=0.0,
            stress=0.0,
            task="tone",
            register="story",
        )
        assert captured["max_tokens"] == 500


class TestValidation:
    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            SpeechController(FakeLlmClient(), f_threshold=-1.0)

    def test_bad_recall_limit_raises(self) -> None:
        with pytest.raises(ValueError):
            SpeechController(FakeLlmClient(), recall_limit=0)

    def test_bad_register_raises(self) -> None:
        with pytest.raises(ValueError):
            SpeechController(FakeLlmClient(), default_register="epic")