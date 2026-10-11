"""Unit tests for CMC projection — resolve_reads / segment_names (Core, ADR-0013).

Core is pure: names → binary mask, fail-fast on unresolvable reads.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.cmc.projection import resolve_reads, segment_names

SEGMENTS = {"circadian": 2, "battery": 1, "cpu": 1, "message": 2}


def test_reads_none_is_full_input() -> None:
    """reads=None → m ≡ 1 (legacy: весь вход)."""
    mask = resolve_reads(None, SEGMENTS)

    np.testing.assert_array_equal(mask, np.ones(6))


def test_reads_subset_mask() -> None:
    """reads подмножества → 1.0 на его срезе, 0.0 на остальных."""
    mask = resolve_reads(("battery", "message"), SEGMENTS)

    # circadian=[0:2], battery=[2], cpu=[3], message=[4:6]
    np.testing.assert_array_equal(mask, [0, 0, 1, 0, 1, 1])


def test_reads_order_independent() -> None:
    """Порядок имён в reads не влияет на маску (строится по карте)."""
    a = resolve_reads(("message", "battery"), SEGMENTS)
    b = resolve_reads(("battery", "message"), SEGMENTS)

    np.testing.assert_array_equal(a, b)


def test_reads_all_segments() -> None:
    """Все сегменты явно → m ≡ 1 (как None)."""
    mask = resolve_reads(tuple(SEGMENTS), SEGMENTS)

    np.testing.assert_array_equal(mask, np.ones(6))


def test_reads_empty_fail_fast() -> None:
    """Пустой reads → ValueError (колонка без входа)."""
    with pytest.raises(ValueError):
        resolve_reads((), SEGMENTS)


def test_reads_unknown_fail_fast() -> None:
    """Неизвестный сегмент → ValueError."""
    with pytest.raises(ValueError):
        resolve_reads(("battery", "banana"), SEGMENTS)


def test_reads_duplicate_fail_fast() -> None:
    """Повтор имени → ValueError."""
    with pytest.raises(ValueError):
        resolve_reads(("battery", "battery"), SEGMENTS)


def test_segment_dims_empty_fail_fast() -> None:
    """Пустая карта сегментов → ValueError."""
    with pytest.raises(ValueError):
        resolve_reads(("battery",), {})


def test_segment_dims_nonpositive_fail_fast() -> None:
    """Неположительная длина сегмента → ValueError."""
    with pytest.raises(ValueError):
        resolve_reads(("battery",), {"battery": 0})


def test_mask_dtype_float64() -> None:
    """Маска — float64 (совместима с арифметикой column_step)."""
    assert resolve_reads(None, SEGMENTS).dtype == np.float64
    assert resolve_reads(("battery",), SEGMENTS).dtype == np.float64


def test_segment_names_reports_reads() -> None:
    """segment_names возвращает сегменты, покрытые маской (наблюдаемость)."""
    mask = resolve_reads(("cpu", "message"), SEGMENTS)

    assert segment_names(mask, SEGMENTS) == ("cpu", "message")


def test_segment_names_full_mask() -> None:
    """Полная маска → все сегменты в порядке карты."""
    mask = resolve_reads(None, SEGMENTS)

    assert segment_names(mask, SEGMENTS) == tuple(SEGMENTS)


def test_segment_names_shape_mismatch() -> None:
    """mask.shape != (Σ dims,) → ValueError."""
    with pytest.raises(ValueError):
        segment_names(np.ones(5), SEGMENTS)
