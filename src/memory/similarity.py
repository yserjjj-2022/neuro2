"""Functional Core — cosine similarity for semantic comparison.

Pure function, no I/O, no SQLite. Tested without a file system.
Used for tests and potential re-ranking of recall() results (Phase 2).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Любой float-вектор: float32, float64 и т.д.
# (serialize_embedding даункастит в float32 на входе,
#  deserialize_embedding возвращает float32 — подтип floating[Any])
Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Косинусная схожесть двух векторов.

    Чистая функция: тестируется без SQLite, без файловой системы.
    Используется для тестов и потенциального re-ranking результатов recall.

    Args:
        a: Первый вектор.
        b: Второй вектор.

    Returns:
        Косинусная схожесть ∈ [-1.0, 1.0].

    Raises:
        ValueError: Если a.shape != b.shape.
        ValueError: Если векторы пустые (size == 0).

    Formula:
        cos(a, b) = dot(a, b) / (||a|| · ||b||)

    Edge cases:
        - ‖a‖ == 0 ИЛИ ‖b‖ == 0 → return 0.0 (деление на ноль в знаменателе)
        - Пустые векторы → ValueError
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: a {a.shape} != b {b.shape}")
    if a.size == 0:
        raise ValueError("Vectors must be non-empty")

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    # Нулевой множитель в знаменателе → схожесть не определена → 0.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))
