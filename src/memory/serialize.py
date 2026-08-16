"""Functional Core — embedding serialization to/from BLOB.

Pure functions, no I/O. BLOB format: float32 little-endian, contiguous.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Vector = np.ndarray[Any, np.dtype[np.float32]]


def serialize_embedding(embedding: Vector) -> bytes:
    """Сериализация эмбеддинга в BLOB для хранения в SQLite.

    Чистая функция: np.ndarray → bytes, без I/O.

    Args:
        embedding: Вектор эмбеддинга.

    Returns:
        BLOB (bytes) для записи в SQLite.

    Note:
        Формат: float32 little-endian, contiguous.
        np.asarray(embedding, dtype=np.float32).tobytes()
    """
    return np.asarray(embedding, dtype=np.float32).tobytes()


def deserialize_embedding(blob: bytes, dim: int) -> Vector:
    """Десериализация BLOB из SQLite в np.ndarray.

    Чистая функция: bytes → np.ndarray, без I/O.

    Args:
        blob: BLOB из SQLite.
        dim: Ожидаемая размерность вектора.

    Returns:
        Вектор эмбеддинга (np.float32).

    Raises:
        ValueError: Если len(blob) != dim * 4 (размер float32).

    Note:
        .copy() обязателен: np.frombuffer возвращает read-only view
        (WRITEABLE=False). Копия стоит почти ничего по производительности
        на типичных размерностях эмбеддингов, но убирает целый класс
        будущих сюрпризов (in-place операции вроде `embedding += delta`
        в Фазе 2 иначе упали бы с ValueError).
    """
    expected = dim * 4  # float32 = 4 байта
    if len(blob) != expected:
        raise ValueError(
            f"Blob length {len(blob)} != expected {expected} (dim={dim} * 4)"
        )
    return np.frombuffer(blob, dtype=np.float32).reshape(dim).copy()
