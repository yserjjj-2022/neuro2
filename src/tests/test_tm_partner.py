"""Unit tests for PartnerModel — the theory-of-mind shell (S5)."""

from __future__ import annotations

import numpy as np

from src.memory.embedder import EmbedderError, FakeEmbedder
from src.tm import PartnerModel, PartnerState


class _BrokenEmbedder:
    """Embedder, который всегда падает (проверка деградации)."""

    dim = 8

    def embed(self, text: str) -> object:
        raise EmbedderError("boom")


class TestPartnerModel:
    def test_first_observation_creates_signature(self) -> None:
        model = PartnerModel(FakeEmbedder(dim=16))
        state = model.observe("привет, как дела?")
        assert len(model.signatures) == 1
        assert state.uncertainty == 1.0  # ничего не было — не узнан
        assert state.name == ""

    def test_similar_utterance_matches(self) -> None:
        model = PartnerModel(FakeEmbedder(dim=16), match_threshold=0.1)
        model.observe("привет мир")
        state = model.observe("привет мир")
        assert len(model.signatures) == 1
        assert state.uncertainty < 1.0

    def test_distinct_style_creates_new_signature(self) -> None:
        model = PartnerModel(FakeEmbedder(dim=32), match_threshold=0.99)
        model.observe("совершенно другой текст один")
        model.observe("абсолютно иная фраза два")
        assert len(model.signatures) == 2

    def test_empty_text_no_change(self) -> None:
        model = PartnerModel(FakeEmbedder(dim=8))
        state = model.observe("")
        assert len(model.signatures) == 0
        assert state.uncertainty == 1.0

    def test_trust_grows_on_agreement(self) -> None:
        model = PartnerModel(FakeEmbedder(dim=16), match_threshold=0.1)
        model.observe("привет")
        state = model.observe("привет")
        assert state.trust > 0.0

    def test_set_name_attaches_to_signature(self) -> None:
        model = PartnerModel(FakeEmbedder(dim=16))
        model.observe("привет")
        model.set_name("Сергей", aliases=("Серёжа",))
        assert model.state.name == "Сергей"
        assert model.signatures[0].name == "Сергей"
        assert model.signatures[0].aliases == ("Серёжа",)

    def test_embedder_failure_degrades_safely(self) -> None:
        model = PartnerModel(_BrokenEmbedder())  # type: ignore[arg-type]
        state = model.observe("привет")
        assert state.uncertainty == 1.0
        assert isinstance(state, PartnerState)

    def test_deterministic(self) -> None:
        a = PartnerModel(FakeEmbedder(dim=16))
        b = PartnerModel(FakeEmbedder(dim=16))
        for text in ("раз", "два", "раз"):
            a.observe(text)
            b.observe(text)
        assert len(a.signatures) == len(b.signatures)
        for sig_a, sig_b in zip(a.signatures, b.signatures):
            np.testing.assert_array_equal(sig_a.centroid, sig_b.centroid)
            assert sig_a.weight == sig_b.weight
