"""Unit tests for precision weighting (inverse_variance + PrecisionEstimator)."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.energy.precision import PrecisionEstimator, inverse_variance


class TestInverseVariance:
    """Чистое ядро γ = 1/var."""

    def test_constant_channel_max_gamma(self) -> None:
        """Константный канал (var=0) → γ = gamma_max."""
        samples = np.ones((10, 2))
        gamma = inverse_variance(samples, eps=1e-6, gamma_max=1e6)
        assert gamma.shape == (2,)
        assert np.all(gamma == pytest.approx(1e6))

    def test_default_gamma_max_is_10(self) -> None:
        """Дефолт gamma_max=10.0 (не 1e6 — иначе взрыв F, см. S1 SPEC)."""
        samples = np.ones((10, 1))
        gamma = inverse_variance(samples)
        assert gamma[0] == pytest.approx(10.0)

    def test_noisy_channel_lower_gamma(self) -> None:
        """Шумный канал → γ ниже, чем у стабильного."""
        stable = np.zeros((100, 1))
        noisy = np.random.default_rng(0).standard_normal((100, 1))
        samples = np.hstack([stable, noisy])
        gamma = inverse_variance(samples)
        assert gamma[0] > gamma[1]

    def test_single_sample(self) -> None:
        """Один сэмпл → var=0 → γ = gamma_max."""
        samples = np.array([[3.0, 4.0]])
        gamma = inverse_variance(samples, gamma_max=100.0)
        assert np.all(gamma == pytest.approx(100.0))

    def test_purity(self) -> None:
        """inverse_variance не мутирует вход."""
        samples = np.array([[1.0], [2.0], [3.0]])
        before = samples.copy()
        inverse_variance(samples)
        np.testing.assert_array_equal(samples, before)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            inverse_variance(np.empty((0, 3)))

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError):
            inverse_variance(np.array([1.0, 2.0]))

    def test_clip_gamma_max(self) -> None:
        """γ не превышает gamma_max."""
        samples = np.ones((5, 1))
        gamma = inverse_variance(samples, gamma_max=10.0)
        assert gamma[0] == pytest.approx(10.0)


class TestPrecisionEstimator:
    """Shell: окно наблюдений."""

    def test_first_update(self) -> None:
        est = PrecisionEstimator(dim=3)
        gamma = est.update(np.array([1.0, 2.0, 3.0]))
        assert gamma.shape == (3,)
        assert est.count == 1

    def test_window_bounded(self) -> None:
        """Число наблюдений не превышает window."""
        est = PrecisionEstimator(dim=2, window=5)
        for i in range(20):
            est.update(np.array([float(i), float(i)]))
        assert est.count == 5

    def test_stable_input_high_gamma(self) -> None:
        est = PrecisionEstimator(dim=1, window=10)
        for _ in range(10):
            gamma = est.update(np.array([1.0]))
        assert gamma[0] == pytest.approx(10.0)

    def test_noisy_input_lower_gamma(self) -> None:
        rng = np.random.default_rng(0)
        est = PrecisionEstimator(dim=1, window=50)
        for _ in range(50):
            gamma = est.update(np.array([rng.standard_normal()]))
        assert gamma[0] < 1e3

    def test_shape_mismatch_raises(self) -> None:
        est = PrecisionEstimator(dim=3)
        with pytest.raises(ValueError):
            est.update(np.array([1.0, 2.0]))

    def test_invalid_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            PrecisionEstimator(dim=0)

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError):
            PrecisionEstimator(dim=1, window=0)

    def test_reset(self) -> None:
        est = PrecisionEstimator(dim=2)
        est.update(np.array([1.0, 1.0]))
        est.reset()
        assert est.count == 0
