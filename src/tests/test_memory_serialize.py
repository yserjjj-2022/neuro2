"""Unit tests for serialize/deserialize embedding — pure core, no SQLite."""

import numpy as np
import pytest

from src.memory.serialize import deserialize_embedding, serialize_embedding


def test_serialize_deserialize_roundtrip() -> None:
    """serialize → deserialize = исходный вектор (с допуском float32)."""
    vec = np.array([0.1, 0.2, -0.3, 1.0], dtype=np.float64)

    blob = serialize_embedding(vec)
    restored = deserialize_embedding(blob, dim=4)

    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, vec, rtol=1e-6)


def test_serialize_deserialize_roundtrip_float32() -> None:
    """float32-вход → round-trip точный."""
    vec = np.array([0.1, 0.2, -0.3, 1.0], dtype=np.float32)

    blob = serialize_embedding(vec)
    restored = deserialize_embedding(blob, dim=4)

    np.testing.assert_array_equal(restored, vec)


def test_serialize_blob_size() -> None:
    """BLOB занимает dim * 4 байт (float32)."""
    vec = np.array([1.0, 2.0, 3.0])
    blob = serialize_embedding(vec)
    assert len(blob) == 3 * 4


def test_deserialize_bad_length() -> None:
    """ValueError при len(blob) != dim * 4."""
    blob = b"\x00" * 16  # 16 байт = dim 4
    with pytest.raises(ValueError):
        deserialize_embedding(blob, dim=5)


def test_deserialize_returns_writable_array() -> None:
    """deserialize возвращает изменяемый массив (не read-only view)."""
    blob = serialize_embedding(np.array([1.0, 2.0, 3.0]))
    restored = deserialize_embedding(blob, dim=3)

    assert restored.flags.writeable
    # In-place операция не должна падать с "assignment destination is read-only"
    restored += 1.0
