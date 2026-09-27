"""Unit tests for the pre-column attention barrier (src/core/cmc/attention.py).

Pure functions: γ (channel trust) controls how much input reaches the columns.
Low γ (noisy) is attenuated; high γ (trusted) passes. The gate is off by
default → identity (S1–S3 contour unchanged).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.cmc.attention import apply_attention, attention_gate


class TestAttentionGate:
    """Tests for attention_gate — γ → weights ∈ [floor, 1)."""

    def test_monotonic_in_gamma(self) -> None:
        """Больше γ → не меньше вес (монотонность)."""
        gamma = np.array([0.1, 1.0, 10.0], dtype=np.float64)
        weights = attention_gate(gamma)
        assert weights[0] < weights[1] < weights[2]

    def test_high_gamma_passes(self) -> None:
        """Очень высокая γ → вес близок к 1."""
        weights = attention_gate(np.array([1e6], dtype=np.float64))
        assert weights[0] == pytest.approx(1.0, abs=1e-4)

    def test_low_gamma_attenuated(self) -> None:
        """Низкая γ → вес близок к floor (барьер)."""
        weights = attention_gate(np.array([1e-6], dtype=np.float64), floor=0.0)
        assert weights[0] < 0.01

    def test_floor_raises_minimum(self) -> None:
        """floor поднимает минимальный вес."""
        weights = attention_gate(np.array([0.0], dtype=np.float64), floor=0.2)
        assert weights[0] == pytest.approx(0.2)

    def test_gamma_ref_midpoint(self) -> None:
        """γ == gamma_ref → середина между floor и 1."""
        weights = attention_gate(np.array([1.0], dtype=np.float64), gamma_ref=1.0)
        assert weights[0] == pytest.approx(0.5)

    def test_weights_bounded(self) -> None:
        """Веса ∈ [floor, 1)."""
        gamma = np.array([1e-9, 1.0, 1e9], dtype=np.float64)
        weights = attention_gate(gamma, floor=0.1)
        assert np.all(weights >= 0.1)
        assert np.all(weights < 1.0)

    def test_purity(self) -> None:
        """Одинаковый вход → одинаковый выход; вход не мутируется."""
        gamma = np.array([0.5, 2.0], dtype=np.float64)
        copy = gamma.copy()
        attention_gate(gamma)
        np.testing.assert_array_equal(gamma, copy)

    def test_invalid_gamma_ref_raises(self) -> None:
        """gamma_ref <= 0 → ValueError."""
        with pytest.raises(ValueError):
            attention_gate(np.array([1.0]), gamma_ref=0.0)

    def test_invalid_floor_raises(self) -> None:
        """floor вне [0, 1) → ValueError."""
        with pytest.raises(ValueError):
            attention_gate(np.array([1.0]), floor=1.0)
        with pytest.raises(ValueError):
            attention_gate(np.array([1.0]), floor=-0.1)


class TestApplyAttention:
    """Tests for apply_attention — u_eff = u · a."""

    def test_identity_when_weights_one(self) -> None:
        """Веса = 1 → вход не изменён."""
        u = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        weights = np.ones(3, dtype=np.float64)
        np.testing.assert_allclose(apply_attention(u, weights), u)

    def test_attenuation(self) -> None:
        """Веса < 1 → вход аттенюирован покомпонентно."""
        u = np.array([2.0, 4.0], dtype=np.float64)
        weights = np.array([0.5, 0.25], dtype=np.float64)
        np.testing.assert_allclose(apply_attention(u, weights), [1.0, 1.0])

    def test_purity(self) -> None:
        """Вход не мутируется."""
        u = np.array([2.0, 4.0], dtype=np.float64)
        copy = u.copy()
        apply_attention(u, np.array([0.5, 0.5]))
        np.testing.assert_array_equal(u, copy)

    def test_shape_mismatch_raises(self) -> None:
        """Несовпадение shapes → ValueError."""
        with pytest.raises(ValueError):
            apply_attention(np.array([1.0, 2.0]), np.array([1.0]))
