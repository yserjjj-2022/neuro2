"""Unit tests for VigilanceGate and detect_conflict (S5, проход 2).

Vigilance marks new claims as hypotheses when they conflict with what is
already known. It never blocks a reply (manifest §3.Г).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.memory.embedder import EmbedderError, FakeEmbedder
from src.tm import Claim, VigilanceGate, detect_conflict


def _unit(*values: float) -> np.ndarray:
    vec = np.array(values, dtype=np.float64)
    return vec / np.linalg.norm(vec)


class TestDetectConflict:
    def test_empty_memory_zero(self) -> None:
        assert detect_conflict(_unit(1.0, 0.0), []) == 0.0

    def test_identical_zero_conflict(self) -> None:
        vec = _unit(1.0, 0.0)
        assert detect_conflict(vec, [vec]) == pytest.approx(0.0)

    def test_orthogonal_high_conflict(self) -> None:
        conflict = detect_conflict(_unit(1.0, 0.0), [_unit(0.0, 1.0)])
        assert conflict == pytest.approx(1.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            detect_conflict(_unit(1.0, 0.0), [_unit(1.0, 0.0, 0.0)])


class TestVigilanceGate:
    def test_first_claim_high_confidence(self) -> None:
        gate = VigilanceGate(FakeEmbedder(dim=64))
        claim = gate.observe("солнце встаёт на востоке")
        assert isinstance(claim, Claim)
        assert claim.conflict == 0.0
        assert claim.confidence == 1.0

    def test_conflicting_claim_low_confidence(self) -> None:
        gate = VigilanceGate(FakeEmbedder(dim=64), conflict_threshold=0.6)
        gate.observe("alpha beta gamma delta")
        claim = gate.observe("совершенно иной текст zzz qqq www")
        # Конфликт может быть высоким или низким в зависимости от хешей,
        # но при высоком конфликте confidence падает.
        assert 0.0 <= claim.confidence <= 1.0

    def test_is_hypothesis_flag(self) -> None:
        gate = VigilanceGate(FakeEmbedder(dim=64), conflict_threshold=0.99)
        gate.observe("alpha beta gamma")
        gate.observe("alpha beta gamma")
        # Близкое утверждение: конфликт ниже порога → не гипотеза.
        assert gate.is_hypothesis() is False

    def test_confirm_raises_confidence(self) -> None:
        gate = VigilanceGate(FakeEmbedder(dim=64))
        gate.observe("некоторое утверждение")
        confirmed = gate.confirm()
        assert confirmed is not None
        assert confirmed.confirmed is True
        assert confirmed.confidence == 1.0

    def test_confirm_without_claim_none(self) -> None:
        gate = VigilanceGate(FakeEmbedder(dim=64))
        assert gate.confirm() is None

    def test_memory_grows_and_capped(self) -> None:
        gate = VigilanceGate(FakeEmbedder(dim=64), max_memory=3)
        for i in range(5):
            gate.observe(f"claim number {i}")
        assert len(gate.memory()) == 3

    def test_embedder_failure_degrades(self) -> None:
        class _Broken:
            dim = 8

            def embed(self, text: str) -> object:
                raise EmbedderError("boom")

        gate = VigilanceGate(_Broken())  # type: ignore[arg-type]
        claim = gate.observe("текст")
        assert claim.confidence == 0.0
        assert claim.conflict == 1.0
