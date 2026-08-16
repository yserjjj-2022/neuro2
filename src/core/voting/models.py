"""Domain objects for voting module.

Frozen dataclass analogous to FreeEnergyResult (energy), Episode (memory),
EnsembleOutput (cmc). Type aliases follow the idiom from src/memory/serialize.py
— numpy generics are invariant, so parameters must be explicitly typed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Любой float-вектор: float32, float64 и т.д.
Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]

# Индексы: np.argsort возвращает платформо-зависимый intp
# (ровно та типизация, что заложена в stub-файлах NumPy).
IndexVector = np.ndarray[Any, np.dtype[np.intp]]


@dataclass(frozen=True)
class VotingResult:
    """Результат голосования: k победителей и one-hot маска.

    Attributes:
        indices: (k,) int — индексы победителей, по убыванию score.
        mask: (N,) float — one-hot маска: 1.0 для победителей, 0.0 иначе.
            float64, а не int: совместимость с soft-WTA (Фаза 2),
            где маска станет весовой.
        scores: (k,) float — оценки победителей, по убыванию.
    """

    indices: IndexVector
    mask: Vector
    scores: Vector
