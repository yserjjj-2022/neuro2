"""Imperative Shell — CMCEnsemble owns column states and aggregates.

Functional Core / Imperative Shell (ADR-0004):
- Core (column_step) — pure function, tested without Shell
- Shell (CMCEnsemble) — owns states, aggregates, computes active
- In prod: step(u) runs every host tick, results feed energy (e(t))
  and telemetry (active).
"""

from __future__ import annotations

import logging

import numpy as np

from .column import column_step
from .models import ColumnConfig, ColumnState, EnsembleOutput, Vector

logger = logging.getLogger(__name__)


class CMCEnsemble:
    """Imperative Shell: владеет состояниями всех колонок.

    Единые размерности всех колонок — инвариант: тензор [N, In, State]
    (конституция §2.1). Проверяется в __init__, не в step.

    Attributes:
        active_threshold: Колонка активна, если ||e(t)||² > threshold.
    """

    def __init__(
        self,
        columns: list[ColumnConfig],
        active_threshold: float = 1e-8,
    ) -> None:
        """Создание ансамбля: валидация, инициализация состояний нулями.

        Args:
            columns: Конфигурации колонок. Все должны иметь одинаковые
                input_dim и state_dim.
            active_threshold: Колонка активна, если ||e(t)||² > threshold.
                Дефолт 1e-8: EMA никогда не сходится к точному 0.0
                (свойство рекуррентного фильтра в float64), поэтому 0.0
                давала бы ложноположительные active_columns при любой
                длительной сходимости.

        Raises:
            ValueError: Если список пуст, или input_dim/state_dim
                различаются между колонками.
        """
        if not columns:
            raise ValueError("columns list must not be empty")

        input_dim = columns[0].input_dim
        state_dim = columns[0].state_dim
        for cfg in columns:
            if cfg.input_dim != input_dim or cfg.state_dim != state_dim:
                raise ValueError(
                    "All columns must share input_dim and state_dim: "
                    f"got {input_dim}/{state_dim} vs {cfg.input_dim}/{cfg.state_dim}"
                )

        self._columns = list(columns)
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.active_threshold = active_threshold
        self._states = [
            ColumnState(
                x=np.zeros(state_dim, dtype=np.float64),
                e=np.zeros(state_dim, dtype=np.float64),
            )
            for _ in columns
        ]
        self._active: int = 0

    def step(self, u: Vector) -> EnsembleOutput:
        """Один тик: прогнать все колонки через column_step, агрегировать.

        Args:
            u: Вход ансамбля L4, shape == (input_dim,). Один вектор,
                общий для всех колонок (в Фазе 1 проекция = весь вектор).

        Returns:
            EnsembleOutput: errors, states, active.

        Raises:
            ValueError: Если u.shape != (input_dim,).
        """
        if u.shape != (self.input_dim,):
            raise ValueError(
                f"Shape mismatch: u {u.shape} != (input_dim,)=({self.input_dim},)"
            )

        states: list[ColumnState] = []
        for cfg, prev in zip(self._columns, self._states, strict=True):
            states.append(column_step(cfg, u, prev))
        self._states = states

        errors = np.stack([s.e for s in states])
        xs = np.stack([s.x for s in states])

        # Точное сравнение с нулём корректно: нулевая ошибка = точный ноль.
        self._active = int(
            sum(1 for s in states if float(np.sum(s.e**2)) > self.active_threshold)
        )

        return EnsembleOutput(errors=errors, states=xs, active=self._active)

    def reset(self) -> None:
        """Сбросить все состояния колонок в нули (x = 0, e = 0)."""
        self._states = [
            ColumnState(
                x=np.zeros(self.state_dim, dtype=np.float64),
                e=np.zeros(self.state_dim, dtype=np.float64),
            )
            for _ in self._columns
        ]
        self._active = 0

    @property
    def active(self) -> int:
        """Число активных колонок после последнего step(). До первого — 0."""
        return self._active
