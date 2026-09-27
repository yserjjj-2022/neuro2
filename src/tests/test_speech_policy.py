"""Unit tests for policy → speech binding (S4).

ChatSession decides the speech action through policy (S4) or falls back to the
S3 should_speak behaviour. The reflex throttle gate blocks the expensive
initiative call while preserving message replies.
"""

from __future__ import annotations

from pathlib import Path

from src.config import (
    HostConfig,
    MemoryConfig,
    PolicyConfig,
)
from src.core.policy import Action, Preferences
from src.host.loop import build_host_loop
from src.speech.chat import ChatSession
from src.speech.controller import SpeechController
from src.speech.history import ConversationHistory
from src.speech.llm import FakeLlmClient


def _session(
    tmp_path: Path,
    inputs: list[str],
    policy: PolicyConfig | None = None,
) -> tuple[ChatSession, list[str]]:
    """Собрать сессию с fake-вводом/выводом и опциональной policy."""
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
        policy=policy,
    )
    return session, output


class TestPolicySpeechBinding:
    """Policy drives the speech goal and speak/silent decision."""

    def test_policy_none_is_s3_behaviour(self, tmp_path: Path) -> None:
        """policy=None → S3-поведение: отвечает на сообщение."""
        session, output = _session(tmp_path, ["привет", "/quit"], policy=None)
        session.run()
        session.loop.close()
        assert len(output) == 1
        assert output[0] != ""

    def test_policy_disabled_is_s3_behaviour(self, tmp_path: Path) -> None:
        """policy.enabled=False → S3-поведение, policy не вызывается."""
        session, output = _session(
            tmp_path,
            ["привет", "/quit"],
            policy=PolicyConfig(enabled=False),
        )
        session.run()
        session.loop.close()
        assert len(output) == 1
        assert output[0] != ""
        assert session.loop.last_policy_trace is None

    def test_policy_respond_on_message(self, tmp_path: Path) -> None:
        """policy включена → на сообщение выбирается RESPOND, есть реплика."""
        session, output = _session(tmp_path, ["привет", "/quit"], policy=PolicyConfig())
        session.run()
        session.loop.close()
        assert len(output) == 1
        assert output[0] != ""
        assert session.loop.last_policy_trace is not None
        assert session.loop.last_policy_trace.chosen is Action.RESPOND

    def test_policy_silent_suppresses_reply(self, tmp_path: Path) -> None:
        """Все триггеры подавлены → SILENT, хост молчит."""
        session, output = _session(
            tmp_path,
            ["привет", "/quit"],
            policy=PolicyConfig(
                preferences=Preferences(
                    respond_to_messages=False,
                    initiative_f_threshold=1e9,
                    homeostatic_alert=False,
                )
            ),
        )
        session.run()
        session.loop.close()
        assert output == ["[хост промолчал]"]
        assert session.loop.last_policy_trace is not None
        assert session.loop.last_policy_trace.chosen is Action.SILENT

    def test_policy_records_trace(self, tmp_path: Path) -> None:
        """Решение policy фиксируется в трассе (explainability)."""
        session, _ = _session(tmp_path, ["привет", "/quit"], policy=PolicyConfig())
        session.run()
        session.loop.close()
        trace = session.loop.last_policy_trace
        assert trace is not None
        assert trace.reason != ""
        assert len(trace.candidates) == 4


class TestThrottleLlmGate:
    """Reflex throttle gate: blocks initiative, keeps message reply."""

    def test_gate_blocks_initiative(self, tmp_path: Path) -> None:
        """Под throttle инициатива (без сообщения) не вызывает LLM."""
        session, _output = _session(tmp_path, ["/quit"], policy=PolicyConfig())
        session.loop.last_throttle = type(session.loop.last_throttle)(
            active=True, llm_gate=True, reason="test"
        )
        goal, allow = session._decide_goal(has_new_message=False)
        session.loop.close()
        assert goal is None
        assert allow is False

    def test_gate_allows_message_reply(self, tmp_path: Path) -> None:
        """Под throttle ответ на сообщение сохраняется (не инициатива)."""
        session, _ = _session(tmp_path, ["/quit"], policy=PolicyConfig())
        session.loop.last_throttle = type(session.loop.last_throttle)(
            active=True, llm_gate=True, reason="test"
        )
        goal, allow = session._decide_goal(has_new_message=True)
        session.loop.close()
        assert allow is True
        assert goal == "respond"
