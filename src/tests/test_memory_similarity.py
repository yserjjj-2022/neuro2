"""Unit tests for cosine_similarity — pure functional core, no SQLite."""

import numpy as np
import pytest

from src.memory.similarity import cosine_similarity


def test_cosine_identical() -> None:
    """Одинаковые векторы → 1.0."""
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal() -> None:
    """Ортогональные векторы → 0.0."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_opposite() -> None:
    """Противоположные векторы → -1.0."""
    v = np.array([1.0, 2.0])
    assert cosine_similarity(v, -v) == pytest.approx(-1.0)


def test_cosine_shape_mismatch() -> None:
    """ValueError при разных размерностях."""
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_cosine_empty() -> None:
    """ValueError при пустых векторах."""
    a = np.array([])
    b = np.array([])
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_cosine_zero_vector() -> None:
    """‖a‖ == 0 ИЛИ ‖b‖ == 0 → 0.0 (не NaN, не inf)."""
    zero = np.array([0.0, 0.0])
    normal = np.array([1.0, 1.0])

    assert cosine_similarity(zero, normal) == pytest.approx(0.0)
    assert cosine_similarity(normal, zero) == pytest.approx(0.0)

    result = cosine_similarity(zero, normal)
    assert not np.isnan(result)
    assert not np.isinf(result)
