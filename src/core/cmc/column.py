"""Functional Core — single-column dynamics (L4 → L5/6 → L2/3).

Pure function: no I/O, no mutable state. The caller (ensemble.py) owns
the state and passes it explicitly, exactly like FreeEnergyCalculator
receives prev_f/prev_stress as parameters.
"""

from __future__ import annotations

from .models import ColumnConfig, ColumnState, Vector


def column_step(
    cfg: ColumnConfig,
    u: Vector,
    prev: ColumnState,
    mask: Vector | None = None,
) -> ColumnState:
    """Один шаг динамики колонки (с проекцией reads — маска, ADR-0013).

    Чистая функция: не мутирует prev, u и mask, не хранит состояние.
    Состояние передаётся явно — вызывающий код (ensemble) владеет им.

    Args:
        cfg: Конфигурация колонки.
        u: Вход L4, shape == (input_dim,).
        prev: Состояние на предыдущем шаге x(t-1), e(t-1).
        mask: Бинарная маска проекции reads, shape == (input_dim,). None →
            весь вход (legacy, m ≡ 1). Маска означает «не наблюдается»:
            нечитаемые координаты не обновляют x и не создают e (ADR-0013 §2).

    Returns:
        Новый ColumnState: x(t), e(t).

    Raises:
        ValueError: Если u.shape != prev.x.shape (или mask.shape != u.shape).

    Formula (mask is None → m ≡ 1):
        x(t) = x(t-1) + alpha · m ⊙ (u(t) − x(t-1))
        e(t) = m ⊙ (u(t) − x(t))
    """
    if u.shape != prev.x.shape:
        raise ValueError(f"Shape mismatch: u {u.shape} != prev.x {prev.x.shape}")
    if mask is not None and mask.shape != u.shape:
        raise ValueError(f"Shape mismatch: mask {mask.shape} != u {u.shape}")

    # Арифметические операции NumPy создают новые массивы —
    # prev, u и mask не мутируются (контракт чистой функции).
    if mask is None:
        x_new = prev.x + cfg.alpha * (u - prev.x)
        e_new = u - x_new
    else:
        # Маска «не наблюдается»: обнуляем и приращение, и ошибку на
        # нечитаемых координатах — ноль не становится наблюдением.
        delta = mask * (u - prev.x)
        x_new = prev.x + cfg.alpha * delta
        e_new = mask * (u - x_new)

    return ColumnState(x=x_new, e=e_new)
