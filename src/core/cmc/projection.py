"""Functional Core — разрешение проекций колонок (reads → маска).

ADR-0013: колонка объявляет **имена** сегментов шины (``reads``), которые
читает; при сборке ансамбля имена разрешаются в бинарную маску полного входа.
Проекция — маска (не подвектор): инвариант единого тензора ``[N, In, State]``
сохранён (CONSTITUTION §2.1).

Семантика маски — «не наблюдается», а не «наблюдается ноль» (ADR-0013 §2):
нечитаемые координаты не обновляют состояние и не создают ошибку (см.
``column_step``). ``reads=None`` → весь вход (``m ≡ 1``, legacy).

Pure functions: no I/O, no state, deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .models import Vector


def resolve_reads(
    reads: tuple[str, ...] | None,
    segment_dims: Mapping[str, int],
) -> Vector:
    """Разрешить именованные reads колонки в бинарную маску полного входа.

    Args:
        reads: Имена сегментов шины, которые читает колонка. None → весь вход.
        segment_dims: Карта входных сегментов ``{имя: длина}`` в порядке
            укладки (полная карта входа ансамбля).

    Returns:
        Бинарная маска shape == (input_dim,), ``input_dim = Σ segment_dims``:
        1.0 на координатах читаемых сегментов, 0.0 на остальных.

    Raises:
        ValueError: Если ``segment_dims`` пуст/содержит неположительные длины;
            ``reads`` пуст (колонка без входа); ``reads`` содержит повтор или
            неизвестный сегмент (fail-fast до первого тика, ADR-0013 §8).

    Note:
        Порядок ``reads`` не влияет на маску: она строится по порядку
        ``segment_dims`` (устойчивость к перестановке объявления).
    """
    if not segment_dims:
        raise ValueError("segment_dims must not be empty")
    for name, dim in segment_dims.items():
        if dim <= 0:
            raise ValueError(f"segment {name!r} dim must be > 0, got {dim}")

    total_dim = sum(segment_dims.values())

    if reads is None:
        return np.ones(total_dim, dtype=np.float64)

    if not reads:
        raise ValueError("reads must not be empty (column without input)")

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in reads:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"duplicate segment(s) in reads: {sorted(set(duplicates))}")

    unknown = [name for name in reads if name not in segment_dims]
    if unknown:
        raise ValueError(f"unknown segment(s) in reads: {unknown}")

    mask = np.zeros(total_dim, dtype=np.float64)
    offset = 0
    for name, dim in segment_dims.items():
        if name in seen:
            mask[offset : offset + dim] = 1.0
        offset += dim
    return mask


def segment_names(
    mask: Vector,
    segment_dims: Mapping[str, int],
) -> tuple[str, ...]:
    """Имена сегментов, полностью покрытых маской (порядок карты).

    Наблюдаемость (ADR-0013 §8): какие сегменты фактически читает колонка.

    Args:
        mask: Бинарная маска shape == (input_dim,).
        segment_dims: Карта входных сегментов ``{имя: длина}``.

    Returns:
        Кортеж имён в порядке ``segment_dims``.

    Raises:
        ValueError: Если mask.shape != (Σ segment_dims,).
    """
    total_dim = sum(segment_dims.values())
    if mask.shape != (total_dim,):
        raise ValueError(f"mask shape {mask.shape} != (input_dim,)=({total_dim},)")

    names: list[str] = []
    offset = 0
    for name, dim in segment_dims.items():
        if float(np.min(mask[offset : offset + dim])) > 0.0:
            names.append(name)
        offset += dim
    return tuple(names)


__all__ = ["resolve_reads", "segment_names"]
