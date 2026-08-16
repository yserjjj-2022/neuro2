"""Functional Core — single-column dynamics (L4 → L5/6 → L2/3).

Pure function: no I/O, no mutable state. The caller (ensemble.py) owns
the state and passes it explicitly, exactly like FreeEnergyCalculator
receives prev_f/prev_stress as parameters.
"""

from __future__ import annotations

from .models import ColumnConfig, ColumnState, Vector


def column_step(cfg: ColumnConfig, u: Vector, prev: ColumnState) -> ColumnState:
    """Один шаг динамики колонки.

    Чистая функция: не мутирует prev и u, не хранит состояние.
    Состояние передаётся явно — вызывающий код (ensemble) владеет им.

    Args:
        cfg: Конфигурация колонки.
        u: Вход L4, shape == (input_dim,).
        prev: Состояние на предыдущем шаге x(t-1), e(t-1).

    Returns:
        Новый ColumnState: x(t), e(t).

    Raises:
        ValueError: Если u.shape != prev.x.shape.

    Formula:
        x(t) = x(t-1) + alpha · (u(t) − x(t-1))
        e(t) = u(t) − x(t)
    """
    if u.shape != prev.x.shape:
        raise ValueError(f"Shape mismatch: u {u.shape} != prev.x {prev.x.shape}")

    # Арифметические операции NumPy создают новые массивы —
    # prev и u не мутируются (контракт чистой функции).
    x_new = prev.x + cfg.alpha * (u - prev.x)
    e_new = u - x_new

    return ColumnState(x=x_new, e=e_new)
