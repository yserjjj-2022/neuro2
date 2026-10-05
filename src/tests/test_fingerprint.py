"""Tests for the behavioral fingerprint Core (S7-A)."""

from __future__ import annotations

from typing import Any

import pytest

from src.host.fingerprint import (
    BehavioralFingerprint,
    FProfile,
    behavioral_fingerprint,
    fingerprint_distance,
    regression_fingerprint,
)


def _row(
    f: float,
    stress: float = 0.0,
    valence: float = 0.0,
    *,
    reflex_tags: str = "",
    active_columns: int = 1,
    spoke: bool = False,
    throttle: bool = False,
    policy_action: str = "",
) -> dict[str, Any]:
    """Построить строку телеметрии с нужными полями."""
    return {
        "free_energy": f,
        "allostatic_stress": stress,
        "valence": valence,
        "reflex_tags": reflex_tags,
        "active_columns": active_columns,
        "latency_ms": 1.0,
        "drift": False,
        "spoke": spoke,
        "throttle": throttle,
        "policy_action": policy_action,
    }


class TestBehavioralFingerprint:
    """Сборка отпечатка из строк телеметрии."""

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            behavioral_fingerprint([])

    def test_missing_key_raises(self) -> None:
        with pytest.raises(ValueError):
            behavioral_fingerprint([{"free_energy": 1.0}])

    def test_deterministic(self) -> None:
        events = [_row(float(i), stress=float(i) / 2) for i in range(5)]
        assert behavioral_fingerprint(events) == behavioral_fingerprint(events)

    def test_profile_and_counts(self) -> None:
        events = [
            _row(1.0, stress=0.5, reflex_tags="battery", active_columns=1),
            _row(3.0, stress=2.0, reflex_tags="resources", active_columns=0),
            _row(2.0, stress=1.0, active_columns=2, spoke=True),
        ]
        fp = behavioral_fingerprint(events)

        assert fp.f_profile.mean == pytest.approx(2.0)
        assert fp.f_profile.max == pytest.approx(3.0)
        assert fp.stress_peaks == pytest.approx(2.0)
        assert fp.reflex_count == 2
        assert fp.resource_alarms == 1
        assert fp.active_fraction == pytest.approx(2 / 3)
        assert fp.talk_rate == pytest.approx(1 / 3)

    def test_rates(self) -> None:
        events = [
            _row(1.0, throttle=True, policy_action="explore"),
            _row(1.0, throttle=True),
            _row(1.0),
            _row(1.0),
        ]
        fp = behavioral_fingerprint(events)
        assert fp.throttle_rate == pytest.approx(0.5)
        assert fp.explore_rate == pytest.approx(0.25)

    def test_metric_lookup(self) -> None:
        fp = behavioral_fingerprint([_row(1.0), _row(2.0, spoke=True)])
        assert fp.metric("f_max") == pytest.approx(2.0)
        assert fp.metric("talk_rate") == pytest.approx(0.5)
        with pytest.raises(KeyError):
            fp.metric("nope")

    def test_rates_bounded(self) -> None:
        fp = behavioral_fingerprint([_row(1.0, spoke=True, throttle=True)])
        for name in ("talk_rate", "throttle_rate", "explore_rate", "active_fraction"):
            assert 0.0 <= fp.metric(name) <= 1.0


class TestFingerprintDistance:
    """Нормированное расстояние между отпечатками."""

    def test_zero_for_equal(self) -> None:
        fp = behavioral_fingerprint([_row(1.0), _row(2.0)])
        assert fingerprint_distance(fp, fp) == pytest.approx(0.0)

    def test_symmetric_and_positive(self) -> None:
        a = behavioral_fingerprint([_row(1.0), _row(1.0)])
        b = behavioral_fingerprint([_row(5.0, spoke=True), _row(5.0)])
        assert fingerprint_distance(a, b) == pytest.approx(fingerprint_distance(b, a))
        assert fingerprint_distance(a, b) > 0.0
        assert fingerprint_distance(a, b) < 1.0

    def test_manual_construction(self) -> None:
        a = BehavioralFingerprint(
            f_profile=FProfile(mean=0.0, std=0.0, max=0.0),
            reflex_count=0,
            stress_peaks=0.0,
            active_fraction=0.0,
            resource_alarms=0,
            talk_rate=0.0,
            throttle_rate=0.0,
            explore_rate=0.0,
        )
        assert fingerprint_distance(a, a) == pytest.approx(0.0)


class TestRegressionFingerprint:
    """Историческая подробная сводка (перенесена из тестов)."""

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            regression_fingerprint([])

    def test_keys_present(self) -> None:
        fp = regression_fingerprint([_row(1.0, valence=2.0)])
        assert "f_final" in fp
        assert "reflex_events" in fp
        assert "latency_p95_ms" in fp

    def test_sign_changes_respect_significance(self) -> None:
        events = [_row(1.0, valence=2.0), _row(1.0, valence=-2.0)]
        assert regression_fingerprint(events)["valence_sign_changes"] == 1.0
