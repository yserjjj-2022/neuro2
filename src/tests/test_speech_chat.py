"""Unit tests for ChatSession — CLI dialogue stand with injected I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import HostConfig, MemoryConfig
from src.host.loop import build_host_loop
from src.speech.chat import ChatSession
from src.speech.controller import SpeechController
from src.speech.history import ConversationHistory
from src.speech.llm import FakeLlmClient


def _session(
    tmp_path: Path,
    inputs: list[str],
) -> tuple[ChatSession, list[str]]:
    """Собрать сессию с fake-вводом/выводом (память: in-memory DB)."""
    config = HostConfig(
        log_path=str(tmp_path / "run.jsonl"),
        memory=MemoryConfig(db_path=str(tmp_path / "mem.db")),
    )
    loop = build_host_loop(config)
    controller = SpeechController(
        llm=FakeLlmClient(),
        memory=loop.memory.store if loop.memory else None,
        embedder=loop.memory.embedder if loop.memory else None,
    )
    output: list[str] = []
    it = iter(inputs)

    def fake_input(prompt: str) -> str:
        try:
            return next(it)
        except StopIteration as exc:
            raise EOFError from exc

    session = ChatSession(
        loop=loop,
        controller=controller,
        history=ConversationHistory(max_turns=10),
        input_fn=fake_input,
        output_fn=output.append,
    )
    return session, output


class TestChatSession:
    def test_reply_printed(self, tmp_path: Path) -> None:
        session, output = _session(tmp_path, ["привет", "/quit"])
        turns = session.run()
        session.loop.close()
        assert turns == 1
        assert len(output) == 1
        assert output[0] != ""

    def test_quit_stops(self, tmp_path: Path) -> None:
        session, _output = _session(tmp_path, ["/quit"])
        assert session.run() == 0
        session.loop.close()

    def test_clear_command(self, tmp_path: Path) -> None:
        session, output = _session(tmp_path, ["привет", "/clear", "/quit"])
        session.run()
        session.loop.close()
        assert any("очищена" in line for line in output)
        assert len(session.history) == 0

    def test_history_grows(self, tmp_path: Path) -> None:
        session, _ = _session(tmp_path, ["привет", "/quit"])
        session.run()
        session.loop.close()
        # user + assistant
        assert len(session.history) == 2

    def test_max_turns(self, tmp_path: Path) -> None:
        session, _ = _session(tmp_path, ["a", "b", "c", "d"])
        assert session.run(max_turns=2) == 2
        session.loop.close()

    def test_eof_stops(self, tmp_path: Path) -> None:
        session, _ = _session(tmp_path, [])
        assert session.run() == 0
        session.loop.close()

    def test_empty_input_skipped(self, tmp_path: Path) -> None:
        session, output = _session(tmp_path, ["", "/quit"])
        assert session.run() == 1
        session.loop.close()
        assert output == []

    def test_invalid_ticks_per_turn_raises(self, tmp_path: Path) -> None:
        session, _ = _session(tmp_path, [])
        with pytest.raises(ValueError):
            ChatSession(
                loop=session.loop,
                controller=session.controller,
                ticks_per_turn=0,
            )
        session.loop.close()