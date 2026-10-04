"""Domain objects for factorization module (S6).

Sparse factorization instead of a single combinatorial POMDP matrix (manifest
§3.Е): independent small factors (mode / partner / task) with their own prior
and likelihood. Frozen dataclasses analogous to FreeEnergyResult (energy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]

_EPS = 1e-12


@dataclass(frozen=True)
class Factor:
    """Независимый дискретный фактор (состояние + приор + наблюдение).

    Attributes:
        name: Имя фактора ("mode", "partner", "task", ...).
        states: Названия состояний (длина == prior/likelihood).
        prior: Нормированное распределение приора.
        likelihood: Правдоподобие наблюдения по состояниям (нормировано).

    Raises:
        ValueError: Если имена пусты/не уникальны или размерности не совпадают.
    """

    name: str
    states: tuple[str, ...]
    prior: Vector
    likelihood: Vector

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.states:
            raise ValueError("states must not be empty")
        if len(set(self.states)) != len(self.states):
            raise ValueError(f"states must be unique, got {self.states}")
        n = len(self.states)
        if self.prior.shape != (n,) or self.likelihood.shape != (n,):
            raise ValueError(
                f"prior/likelihood shape must be ({n},), got "
                f"{self.prior.shape}/{self.likelihood.shape}"
            )
        if float(np.min(self.prior)) < 0.0 or float(np.min(self.likelihood)) < 0.0:
            raise ValueError("prior/likelihood must be non-negative")


@dataclass(frozen=True)
class FactorizedState:
    """Совместное состояние как произведение независимых факторов.

    Совместное распределение не материализуется (sparse factorization): факторы
    независимы, обновление — покомпонентное (манифест §3.Е).

    Attributes:
        factors: Независимые факторы (уникальные имена).

    Raises:
        ValueError: Если имена не уникальны или список пуст.
    """

    factors: tuple[Factor, ...]

    def __post_init__(self) -> None:
        if not self.factors:
            raise ValueError("factors must not be empty")
        names = [f.name for f in self.factors]
        if len(set(names)) != len(names):
            raise ValueError(f"factor names must be unique, got {names}")


def _normalize(vec: Vector) -> Vector:
    """Нормировать вектор в распределение (равномерное при нулевой сумме)."""
    total = float(np.sum(vec))
    if total <= _EPS:
        return np.full(vec.shape, 1.0 / vec.shape[0], dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / total
