"""Functional Core — pre-column attention barrier (γ, S4).

Manifest §2 / ADR-0005 §2: precision γ is a *vigilance barrier* **before** the
columns, not just a post-hoc error weight. Channel trust (γ) controls how much
of the input reaches the columns: a noisy/untrusted channel (low γ) is
attenuated; a trusted channel (high γ) passes through.

Pure function: no I/O, no state. Applied in the host loop before
``pipeline.tick`` when ``attention_gate`` is enabled.
"""

from __future__ import annotations

import numpy as np

from .models import Vector


def attention_gate(
    gamma: Vector,
    gamma_ref: float = 1.0,
    floor: float = 0.0,
) -> Vector:
    """Веса пред-колоночного барьера внимания из precision γ.

    Формула:
        a(γ) = floor + (1 - floor) · γ / (γ + gamma_ref)

    Семантика:
        - γ → 0 (шумный канал): a → floor (вход аттенюируется — барьер).
        - γ = gamma_ref: a = floor + (1 - floor)/2.
        - γ → ∞ (доверенный канал): a → 1 (вход проходит).
        - Монотонно возрастает по γ; ∈ [floor, 1).

    Args:
        gamma: Вектор precision γ по каналам, shape (D,), γ > 0.
        gamma_ref: Опорная γ (полувысота барьера), > 0.
        floor: Минимальный вес (0 = полное подавление шумного канала), [0, 1).

    Returns:
        Веса attention shape (D,), ∈ [floor, 1).

    Raises:
        ValueError: Если gamma_ref <= 0 или floor вне [0, 1).
    """
    if gamma_ref <= 0.0:
        raise ValueError(f"gamma_ref must be > 0, got {gamma_ref}")
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"floor must be in [0, 1), got {floor}")

    ratio = gamma / (gamma + gamma_ref)
    return floor + (1.0 - floor) * ratio


def apply_attention(u: Vector, weights: Vector) -> Vector:
    """Применить барьер внимания к входу: ``u_eff = u · a``.

    Чистая функция: не мутирует входы, возвращает новый массив.

    Args:
        u: Вход шины, shape (D,).
        weights: Веса attention, shape (D,).

    Returns:
        Аттенюированный вход, shape (D,).

    Raises:
        ValueError: Если shapes не совпадают.
    """
    if u.shape != weights.shape:
        raise ValueError(f"Shape mismatch: u {u.shape} != weights {weights.shape}")
    return np.asarray(u * weights, dtype=np.float64)
