"""Unit tests for voting Functional Core — kwta().

Core is pure: tested without manager, without state, without I/O.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.voting.kwta import kwta


def test_kwta_basic() -> None:
    """Top-k корректно выбирает k наибольших оценок."""
    scores = np.array([0.1, 0.9, 0.4, 0.7])

    result = kwta(scores, k=2)

    assert len(result.indices) == 2
    # Индексы победителей: 1 (0.9) и 3 (0.7)
    assert result.indices.tolist() == [1, 3]
    np.testing.assert_allclose(result.scores, [0.9, 0.7])


def test_kwta_topk_property() -> None:
    """Top-k свойство: ни один проигравший не имеет оценку выше победителя."""
    rng = np.random.default_rng(42)
    scores = rng.random(10)

    result = kwta(scores, k=3)

    winners = result.scores
    losers = np.setdiff1d(np.arange(scores.size), result.indices)
    assert float(winners.min()) >= float(scores[losers].max())


def test_kwta_k_equals_one() -> None:
    """k=1 → один победитель (hard-WTA — вырожденный случай k-WTA)."""
    scores = np.array([0.2, 0.8, 0.5])

    result = kwta(scores, k=1)

    assert len(result.indices) == 1
    assert result.indices[0] == 1
    assert result.mask.sum() == 1


def test_kwta_k_equals_n() -> None:
    """k=N → все колонки победители, mask = 1.0."""
    scores = np.array([0.1, 0.5, 0.3])

    result = kwta(scores, k=3)

    assert len(result.indices) == 3
    np.testing.assert_allclose(result.mask, np.ones(3))
    assert result.mask.sum() == 3


def test_kwta_ties() -> None:
    """Равные scores → побеждает меньший индекс (детерминизм)."""
    scores = np.array([0.5, 0.5, 0.5])

    result = kwta(scores, k=2)

    assert result.indices.tolist() == [0, 1]
    np.testing.assert_allclose(result.mask, [1.0, 1.0, 0.0])


def test_kwta_invalid_k_low() -> None:
    """k < 1 → ValueError (fail-fast)."""
    scores = np.array([0.1, 0.2])

    with pytest.raises(ValueError):
        kwta(scores, k=0)
    with pytest.raises(ValueError):
        kwta(scores, k=-1)


def test_kwta_invalid_k_high() -> None:
    """k > N → ValueError (условие корректности постановки задачи)."""
    scores = np.array([0.1, 0.2])

    with pytest.raises(ValueError):
        kwta(scores, k=3)


def test_kwta_empty_scores() -> None:
    """Пустые scores → ValueError."""
    with pytest.raises(ValueError):
        kwta(np.array([]), k=1)


def test_kwta_2d_scores() -> None:
    """2D scores → ValueError."""
    scores = np.array([[0.1, 0.2], [0.3, 0.4]])

    with pytest.raises(ValueError):
        kwta(scores, k=1)


def test_kwta_purity() -> None:
    """Чистота: вход не мутируется, повторный вызов → тот же результат."""
    scores = np.array([0.1, 0.9, 0.4])
    original = scores.copy()

    r1 = kwta(scores, k=2)
    r2 = kwta(scores, k=2)

    np.testing.assert_array_equal(scores, original)  # вход не изменён
    np.testing.assert_array_equal(r1.indices, r2.indices)
    np.testing.assert_array_equal(r1.mask, r2.mask)

    # Мутация входного массива после вызова не влияет на результат
    scores[:] = 999.0
    np.testing.assert_array_equal(r1.scores, [0.9, 0.4])


def test_kwta_mask_sum() -> None:
    """Маска содержит ровно k единиц: mask.sum() == k."""
    rng = np.random.default_rng(7)
    scores = rng.random(8)

    for k in (1, 3, 8):
        result = kwta(scores, k=k)
        assert result.mask.sum() == k
        assert len(result.indices) == k
