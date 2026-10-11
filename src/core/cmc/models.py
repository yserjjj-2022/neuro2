"""Domain objects for CMC module.

Frozen dataclasses analogous to FreeEnergyResult (energy) and
TelemetryEvent (telemetry). Vector type alias follows the idiom from
src/memory/serialize.py — numpy generics are invariant, so
np.floating[Any] accepts both float32 and float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Любой float-вектор: float32, float64 и т.д.
# (column_step принимает любой float-вход, результат — тот же dtype)
Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]


@dataclass(frozen=True)
class ColumnConfig:
    """Конфигурация одной колонки. Только параметры — состояние не здесь.

    Attributes:
        input_dim: Размерность входа L4.
        state_dim: Размерность состояния L5/6. В Фазе 1 == input_dim.
        specialization: Тег колонки ("tone", "rhythm", "meaning", ...) —
            идентификатор специализации в выборе задачи (k-WTA по тегу).
        alpha: Скорость обновления состояния, α ∈ [0, 1].
            x(t) = x(t-1) + α·m⊙(u(t) − x(t-1)). Дефолт 0.1 (≈10 тиков).
        reads: Имена сегментов шины, которые читает колонка (ADR-0013).
            None → весь вход (legacy, m ≡ 1). Пустой кортеж → ValueError
            (колонка без входа). Имена разрешаются в бинарную маску при сборке
            ансамбля (Shell), не здесь. Проекция — маска полного вектора, не
            подвектор (инвариант [N, In, State], CONSTITUTION §2.1).
    """

    input_dim: int
    state_dim: int
    specialization: str = "general"
    alpha: float = 0.1
    reads: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be > 0, got {self.state_dim}")
        if self.state_dim != self.input_dim:
            raise ValueError(
                f"state_dim ({self.state_dim}) must equal input_dim "
                f"({self.input_dim}) in Phase 1 — no projection matrices"
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.reads is not None:
            if not self.reads:
                raise ValueError("reads must not be empty (column without input)")
            if any(not name for name in self.reads):
                raise ValueError("reads segment names must not be empty")


@dataclass(frozen=True)
class ColumnState:
    """Иммутабельный снимок состояния колонки: x(t) и e(t).

    Аналог FreeEnergyResult (energy) и Episode (memory) — frozen dataclass
    как снимок состояния в момент t.

    Attributes:
        x: Состояние L5/6, shape == (state_dim,).
        e: Ошибка предсказания L2/3, shape == (state_dim,).
    """

    x: Vector
    e: Vector


@dataclass(frozen=True)
class EnsembleOutput:
    """Снимок всего ансамбля после step().

    Attributes:
        errors: e(t) всех колонок, shape == (N, input_dim).
        states: x(t) всех колонок, shape == (N, state_dim).
        active: Число активных колонок (||e||² > threshold).
    """

    errors: Vector
    states: Vector
    active: int
