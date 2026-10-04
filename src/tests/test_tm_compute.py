"""Unit tests for the theory-of-mind functional core (S5).

Pure functions: no SQLite, no network. Vectors are built directly so the tests
exercise matching/update semantics, not the embedder.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tm.compute import (
    match_partner,
    normalize_pause,
    update_signature,
    update_trust,
)
from src.tm.models import PartnerSignature, PartnerState


def _unit(*values: float) -> np.ndarray:
    vec = np.array(values, dtype=np.float64)
    return vec / np.linalg.norm(vec)


def _signature(centroid: np.ndarray, **kwargs: object) -> PartnerSignature:
    return PartnerSignature(centroid=centroid, **kwargs)  # type: ignore[arg-type]


class TestMatchPartner:
    def test_empty_signatures_returns_none(self) -> None:
        index, sim = match_partner(_unit(1.0, 0.0), [], threshold=0.5)
        assert index is None
        assert sim == 0.0

    def test_matches_above_threshold(self) -> None:
        sig = _signature(_unit(1.0, 0.0))
        index, sim = match_partner(_unit(1.0, 0.0), [sig], threshold=0.75)
        assert index == 0
        assert sim == pytest.approx(1.0)

    def test_below_threshold_not_matched(self) -> None:
        sig = _signature(_unit(1.0, 0.0))
        index, _ = match_partner(_unit(0.0, 1.0), [sig], threshold=0.75)
        assert index is None

    def test_picks_best_of_many(self) -> None:
        sigs = [
            _signature(_unit(1.0, 0.0)),
            _signature(_unit(0.9, 0.1)),
        ]
        index, _ = match_partner(_unit(0.9, 0.1), sigs, threshold=0.5)
        assert index == 1

    def test_deterministic(self) -> None:
        sigs = [_signature(_unit(1.0, 0.0))]
        assert match_partner(_unit(1.0, 0.0), sigs, threshold=0.5) == match_partner(
            _unit(1.0, 0.0), sigs, threshold=0.5
        )

    def test_shape_mismatch_raises(self) -> None:
        sig = _signature(_unit(1.0, 0.0))
        with pytest.raises(ValueError):
            match_partner(_unit(1.0, 0.0, 0.0), [sig], threshold=0.5)

    def test_bad_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            match_partner(_unit(1.0, 0.0), [], threshold=1.5)


class TestUpdateSignature:
    def test_creates_from_none(self) -> None:
        sig = update_signature(
            None, _unit(1.0, 0.0), pause_s=2.0, valence=0.3, stress=0.4,
            learning_rate=0.2,
        )
        assert sig.weight == 1.0
        assert sig.mean_pause_s == pytest.approx(2.0)
        assert sig.mean_valence == pytest.approx(0.3)
        assert sig.mean_stress == pytest.approx(0.4)

    def test_moving_average_converges(self) -> None:
        target = _unit(0.0, 1.0)
        sig = update_signature(
            None, _unit(1.0, 0.0), pause_s=0.0, valence=0.0, stress=0.0,
            learning_rate=0.5,
        )
        for _ in range(20):
            sig = update_signature(
                sig, target, pause_s=0.0, valence=0.0, stress=0.0,
                learning_rate=0.5,
            )
        cosine = float(np.dot(sig.centroid, target) / np.linalg.norm(sig.centroid))
        assert cosine == pytest.approx(1.0, abs=1e-3)

    def test_purity_input_unchanged(self) -> None:
        sig = _signature(_unit(1.0, 0.0), weight=3.0)
        before = sig.centroid.copy()
        update_signature(
            sig, _unit(0.0, 1.0), pause_s=1.0, valence=0.0, stress=0.0,
            learning_rate=0.5,
        )
        np.testing.assert_array_equal(sig.centroid, before)

    def test_bad_learning_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            update_signature(
                None, _unit(1.0, 0.0), pause_s=0.0, valence=0.0, stress=0.0,
                learning_rate=0.0,
            )

    def test_negative_pause_raises(self) -> None:
        with pytest.raises(ValueError):
            update_signature(
                None, _unit(1.0, 0.0), pause_s=-1.0, valence=0.0, stress=0.0,
                learning_rate=0.5,
            )


class TestUpdateTrust:
    def test_agreement_raises_trust(self) -> None:
        state = PartnerState(trust=0.5)
        new = update_trust(
            state, conflict=0.0, ambiguity=0.0, trust_gain=0.1, trust_decay=0.0
        )
        assert new.trust == pytest.approx(0.6)

    def test_conflict_lowers_trust(self) -> None:
        state = PartnerState(trust=0.5)
        new = update_trust(
            state, conflict=1.0, ambiguity=0.0, trust_gain=0.1, trust_decay=0.05
        )
        assert new.trust == pytest.approx(0.45)

    def test_clamped_to_bounds(self) -> None:
        state = PartnerState(trust=0.99)
        new = update_trust(
            state, conflict=0.0, ambiguity=0.0, trust_gain=1.0, trust_decay=0.0
        )
        assert new.trust == 1.0
        state_low = PartnerState(trust=0.01)
        low = update_trust(
            state_low, conflict=1.0, ambiguity=0.0, trust_gain=0.0, trust_decay=1.0
        )
        assert low.trust == 0.0

    def test_bad_conflict_raises(self) -> None:
        with pytest.raises(ValueError):
            update_trust(
                PartnerState(), conflict=1.5, ambiguity=0.0,
                trust_gain=0.1, trust_decay=0.0,
            )


class TestNormalizePause:
    def test_zero_pause_is_zero(self) -> None:
        assert normalize_pause(0.0, 5.0) == 0.0

    def test_monotonic(self) -> None:
        assert normalize_pause(1.0, 5.0) < normalize_pause(5.0, 5.0)

    def test_in_range(self) -> None:
        for pause in (0.0, 0.5, 5.0, 100.0):
            assert 0.0 <= normalize_pause(pause, 5.0) < 1.0

    def test_bad_tau_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_pause(1.0, 0.0)
