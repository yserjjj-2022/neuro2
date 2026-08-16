"""Functional Core — k-WTA lateral inhibition.

Pure function: no I/O, no mutable state, input arrays not mutated.
Top-k in one pass — equivalent to the stationary state of a recurrent
lateral inhibition network (winners suppress the rest to zero).
"""

from __future__ import annotations

import numpy as np

from .models import IndexVector, Vector, VotingResult


def kwta(scores: Vector, k: int) -> VotingResult:
    """Чистая k-WTA: выбрать top-k по scores, вернуть индексы + one-hot маску.

    Чистая функция: не мутирует scores, не хранит состояние.
    Одинаковый вход → одинаковый выход (stable ties).

    Args:
        scores: Вектор оценок колонок, shape (N,).
        k: Число победителей, 1 ≤ k ≤ N.

    Returns:
        VotingResult: indices (k,), mask (N,), scores (k,) по убыванию.

    Raises:
        ValueError: Если scores пустой, не одномерный, или k вне [1, N].

    Formula:
        winners = argsort(-scores, kind="stable")[:k]
        mask[winners] = 1.0
    """
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-dimensional, got {scores.ndim}D")
    n = scores.size
    if n == 0:
        raise ValueError("scores must not be empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        raise ValueError(f"k ({k}) must be <= number of scores ({n})")

    # stable-сортировка: при равенстве scores побеждает меньший индекс
    indices: IndexVector = np.argsort(-scores, kind="stable")[:k]

    mask = np.zeros(n, dtype=np.float64)
    mask[indices] = 1.0

    # .copy() обязателен: scores[indices] — view на входной массив
    winner_scores = scores[indices].copy()

    return VotingResult(indices=indices, mask=mask, scores=winner_scores)
