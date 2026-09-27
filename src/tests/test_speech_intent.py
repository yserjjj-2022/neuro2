"""Unit tests for Intent-Frame + registers — pure core, no I/O."""

from __future__ import annotations

import pytest

from src.speech.intent import (
    REGISTER_MAX_TOKENS,
    build_intent_frame,
    describe_affect,
    register_max_tokens,
    render_messages,
)


class TestDescribeAffect:
    def test_positive(self) -> None:
        assert "позитивное" in describe_affect(5.0, 0.0)

    def test_negative(self) -> None:
        assert "негативное" in describe_affect(-5.0, 0.0)

    def test_neutral_band(self) -> None:
        assert "нейтральное" in describe_affect(0.5, 0.0)

    def test_high_stress(self) -> None:
        assert "сильно напряжённое" in describe_affect(0.0, 10.0)

    def test_deterministic(self) -> None:
        assert describe_affect(1.0, 2.0) == describe_affect(1.0, 2.0)


class TestRegister:
    def test_max_tokens_ordering(self) -> None:
        assert register_max_tokens("terse") < register_max_tokens("brief")
        assert register_max_tokens("brief") < register_max_tokens("normal")
        assert register_max_tokens("normal") < register_max_tokens("story")

    def test_unknown_register_raises(self) -> None:
        with pytest.raises(ValueError):
            register_max_tokens("epic")

    def test_default_is_brief(self) -> None:
        assert "brief" in REGISTER_MAX_TOKENS


class TestBuildIntentFrame:
    def test_fields_from_state(self) -> None:
        frame = build_intent_frame(
            f=3.0, valence=-2.0, stress=6.0, task="tone", precedents=("x",)
        )
        assert frame.free_energy == 3.0
        assert frame.valence == -2.0
        assert frame.stress == 6.0
        assert frame.task == "tone"
        assert frame.precedents == ("x",)
        assert frame.goal == "respond"

    def test_affect_derived(self) -> None:
        frame = build_intent_frame(f=0.0, valence=-2.0, stress=0.0, task="t")
        assert "негативное" in frame.affect

    def test_invalid_register_raises(self) -> None:
        with pytest.raises(ValueError):
            build_intent_frame(
                f=0.0, valence=0.0, stress=0.0, task="t", register="epic"
            )

    def test_default_register(self) -> None:
        frame = build_intent_frame(f=0.0, valence=0.0, stress=0.0, task="t")
        assert frame.register == "brief"


class TestRenderMessages:
    def _frame(self, register: str = "brief", precedents: tuple = ()):
        return build_intent_frame(
            f=1.0,
            valence=-1.0,
            stress=2.0,
            task="tone",
            precedents=precedents,
            register=register,
        )

    def test_structure(self) -> None:
        messages = render_messages(self._frame(), "привет")
        assert messages[0]["role"] == "system"
        assert messages[-1] == {"role": "user", "content": "привет"}

    def test_system_contains_state(self) -> None:
        messages = render_messages(self._frame(), "привет")
        system = messages[0]["content"]
        assert "tone" in system
        assert "valence=" in system

    def test_register_hint_in_system(self) -> None:
        brief = render_messages(self._frame("brief"), "x")[0]["content"]
        terse = render_messages(self._frame("terse"), "x")[0]["content"]
        assert brief != terse
        assert "словом" in terse

    def test_precedents_included(self) -> None:
        messages = render_messages(self._frame(precedents=("прошлый эпизод",)), "x")
        assert "прошлый эпизод" in messages[0]["content"]

    def test_no_precedents_no_block(self) -> None:
        messages = render_messages(self._frame(), "x")
        assert "прецеденты" not in messages[0]["content"]

    def test_history_included(self) -> None:
        history = [
            {"role": "user", "content": "старое"},
            {"role": "assistant", "content": "ответ"},
        ]
        messages = render_messages(self._frame(), "новое", history=history)
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]
        assert messages[-1]["content"] == "новое"

    def test_deterministic(self) -> None:
        a = render_messages(self._frame(), "x")
        b = render_messages(self._frame(), "x")
        assert a == b