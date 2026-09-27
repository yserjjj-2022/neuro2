"""Unit tests for the chat status line (pure Core)."""

from __future__ import annotations

from src.speech.status import format_status


class TestFormatStatus:
    def _fmt(self, **overrides: object) -> str:
        base = {
            "f": 2.31,
            "valence": -1.4,
            "stress": 0.8,
            "gamma": 4.2,
            "task": "tone",
            "recall_hit": True,
            "drift": False,
        }
        base.update(overrides)
        return format_status(**base)  # type: ignore[arg-type]

    def test_contains_all_fields(self) -> None:
        line = self._fmt()
        for token in ("F=2.31", "val=-1.40", "stress=0.80", "γ=4.20", "tone"):
            assert token in line

    def test_recall_flags(self) -> None:
        assert "recall=1" in self._fmt(recall_hit=True)
        assert "recall=0" in self._fmt(recall_hit=False)

    def test_drift_flags(self) -> None:
        assert "дрейф=да" in self._fmt(drift=True)
        assert "дрейф=нет" in self._fmt(drift=False)

    def test_signed_valence(self) -> None:
        assert "val=+1.00" in self._fmt(valence=1.0)

    def test_deterministic(self) -> None:
        assert self._fmt() == self._fmt()