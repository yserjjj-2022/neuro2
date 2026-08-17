"""Domain objects for attractors module.

Frozen dataclass analogous to FreeEnergyResult (energy), VotingResult (voting),
EnsembleOutput (cmc). Type aliases follow the idiom from src/memory/serialize.py
— numpy generics are invariant, so parameters must be explicitly typed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Любой float-вектор: float32, float64 и т.д.
Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]


@dataclass(frozen=True)
class TaskAttraction:
    """Снимок аттрактора задачи после tick().

    Аналог FreeEnergyResult (energy) и VotingResult (voting) — frozen
    dataclass как снимок состояния, неизменяемый после возврата.

    Attributes:
        mask: (N,) float64 — текущий аттрактор: one-hot маска активных колонок.
            Устойчив к шуму в scores — меняется только при преодолении
            порога переключения (см. Инвариант 3 SPEC).
        history_size: int — сколько тиков подряд аттрактор не менялся.
            Используется для расчёта min_dwell (гистерезис + STP).
        scores: (N,) float64 — полный вектор активностей всех колонок.
            Копия из текущего входа для отладки и анализа конкурентности.
        converged: bool — True, если scores sufficiently small (EMA сходимость).
            При converged=True: переключение запрещено до history_size >= min_dwell.
    """

    mask: Vector
    history_size: int
    scores: Vector
    converged: bool
