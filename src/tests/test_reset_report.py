"""Tests for reset self-report + report_reset intent + gate (S7-D)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_preset
from src.core.selfcontrol import (
    ChangeKind,
    ResetLevel,
    ResetReport,
    reset_self_report,
)
from src.host.gate import Capability, CapabilityGate
from src.host.loop import build_host_loop
from src.host.sensitivity import DeterministicMeter
from src.speech.chat import ChatSession
from src.speech.intent import report_reset_intent


class TestResetSelfReport:
    """Core: честный самоотчёт о сбросе."""

    def test_no_reset(self) -> None:
        report = reset_self_report(reset_level="")
        assert report.triggered is False
        assert report.certain is True
        assert "не было" in report.text

    @pytest.mark.parametrize("level", ["soft", "freeze", "hard"])
    def test_known_levels(self, level: str) -> None:
        report = reset_self_report(reset_level=level)
        assert report.triggered is True
        assert report.certain is True
        assert report.level == level

    def test_hard_is_not_norm(self) -> None:
        report = reset_self_report(reset_level=ResetLevel.HARD)
        assert "не «норма»" in report.text

    def test_unknown_level_fails_safe(self) -> None:
        report = reset_self_report(reset_level="bogus")
        assert report.certain is False
        assert report.triggered is False
        assert "выдумывать" in report.text

    def test_change_kind_appended(self) -> None:
        report = reset_self_report(
            reset_level="soft", change_kind=ChangeKind.DEVELOPMENT
        )
        assert "development" in report.text

    def test_reason_appended(self) -> None:
        report = reset_self_report(reset_level="freeze", reason="CSD warning")
        assert "CSD warning" in report.text
        assert report.reason == "CSD warning"

    def test_report_frozen(self) -> None:
        import dataclasses

        report = reset_self_report(reset_level="soft")
        assert isinstance(report, ResetReport)
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.text = "x"  # type: ignore[misc]


class TestReportResetIntent:
    """Core: распознавание запроса о сбросе."""

    @pytest.mark.parametrize(
        "text",
        ["что с тобой было?", "ты сбрасывался?", "расскажи про сброс", "report reset"],
    )
    def test_detected(self, text: str) -> None:
        assert report_reset_intent(text) is True

    @pytest.mark.parametrize("text", ["привет", "как дела?", "расскажи анекдот"])
    def test_not_detected(self, text: str) -> None:
        assert report_reset_intent(text) is False


class _StubController:
    """Заглушка SpeechController: не ходит в LLM."""

    def respond(self, **_kwargs: object) -> str:
        return "обычный ответ"


def _session(
    tmp_path: Path, gate: CapabilityGate, out: list[str]
) -> ChatSession:
    import dataclasses

    config = dataclasses.replace(
        load_preset("baseline"), log_path=str(tmp_path / "run.jsonl")
    )
    loop = build_host_loop(config, meter=DeterministicMeter())
    return ChatSession(
        loop,
        _StubController(),  # type: ignore[arg-type]
        input_fn=lambda _prompt: "что с тобой было?",
        output_fn=out.append,
        ticks_per_turn=1,
        policy=None,
        gate=gate,
    )


class TestChatResetReport:
    """Shell: самоотчёт исполняется только через CapabilityGate."""

    def test_spoken_when_allowed(self, tmp_path: Path) -> None:
        out: list[str] = []
        session = _session(
            tmp_path, CapabilityGate(granted=frozenset({Capability.SPEAK})), out
        )
        session.run(max_turns=1)
        assert any("Сбросов не было" in line for line in out)
        assert session.history.as_messages()

    def test_denied_fail_safe(self, tmp_path: Path) -> None:
        out: list[str] = []
        session = _session(tmp_path, CapabilityGate(granted=frozenset()), out)
        session.run(max_turns=1)
        assert out == ["[хост промолчал]"]

    def test_normal_text_uses_controller(self, tmp_path: Path) -> None:
        out: list[str] = []
        config = load_preset("baseline")
        loop = build_host_loop(config, meter=DeterministicMeter())
        session = ChatSession(
            loop,
            _StubController(),  # type: ignore[arg-type]
            input_fn=lambda _p: "привет",
            output_fn=out.append,
            ticks_per_turn=1,
            policy=None,
            gate=CapabilityGate(granted=frozenset({Capability.SPEAK})),
        )
        session.run(max_turns=1)
        assert out == ["обычный ответ"]
