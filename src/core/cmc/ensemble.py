"""Imperative Shell — CMCEnsemble owns column states and aggregates.

Functional Core / Imperative Shell (ADR-0004):
- Core (column_step) — pure function, tested without Shell
- Shell (CMCEnsemble) — owns states, aggregates, computes active
- In prod: step(u) runs every host tick, results feed energy (e(t))
  and telemetry (active).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from .column import column_step
from .models import ColumnConfig, ColumnState, EnsembleOutput, Vector
from .projection import resolve_reads, segment_names

logger = logging.getLogger(__name__)


class CMCEnsemble:
    """Imperative Shell: владеет состояниями всех колонок.

    Единые размерности всех колонок — инвариант: тензор [N, In, State]
    (конституция §2.1). Проверяется в __init__, не в step.

    Проекции (ADR-0013): reads каждой колонки разрешаются в бинарную маску
    полного входа при сборке; маски неизменны в прогоне.

    Attributes:
        active_threshold: Колонка активна, если ||e(t)||² > threshold.
    """

    def __init__(
        self,
        columns: list[ColumnConfig],
        active_threshold: float = 1e-8,
        segment_dims: Mapping[str, int] | None = None,
    ) -> None:
        """Создание ансамбля: валидация, разрешение reads → маски, init состояний.

        Args:
            columns: Конфигурации колонок. Все должны иметь одинаковые
                input_dim и state_dim.
            active_threshold: Колонка активна, если ||e(t)||² > threshold.
                Дефолт 1e-8: EMA никогда не сходится к точному 0.0
                (свойство рекуррентного фильтра в float64), поэтому 0.0
                давала бы ложноположительные active_columns при любой
                длительной сходимости.
            segment_dims: Карта входных сегментов ансамбля ``{имя: длина}`` в
                порядке укладки — для разрешения ``reads`` (ADR-0013). None →
                ``reads`` у колонок должны быть None.

        Raises:
            ValueError: Если список пуст; input_dim/state_dim различаются;
                или ``reads`` не разрешаются (неизвестный сегмент, пустой
                reads, повтор имени, reads без segment_dims).
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

        # Разрешение проекций: reads → бинарные маски (ADR-0013 §8, fail-fast).
        has_reads = any(cfg.reads is not None for cfg in columns)
        if has_reads and segment_dims is None:
            raise ValueError(
                "segment_dims is required when columns declare reads (ADR-0013)"
            )
        self._segment_dims: dict[str, int] = dict(segment_dims or {})
        self._masks: list[Vector | None] = []
        for cfg in columns:
            if segment_dims is None:
                self._masks.append(None)
                continue
            mask = resolve_reads(cfg.reads, segment_dims)
            if mask.shape != (input_dim,):
                raise ValueError(
                    f"reads of column {cfg.specialization!r} resolve to "
                    f"{mask.shape}, expected ({input_dim},)"
                )
            # reads=None → m ≡ 1; храним None, чтобы column_step шёл legacy-ветвью.
            self._masks.append(None if cfg.reads is None else mask)

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
        """Один тик: прогнать все колонки через column_step (со своей маской).

        Args:
            u: Вход ансамбля L4, shape == (input_dim,). Один вектор, общий для
                всех колонок; каждая читает свою проекцию reads (маска полного
                вектора, ADR-0013).

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
        for cfg, prev, mask in zip(
            self._columns, self._states, self._masks, strict=True
        ):
            states.append(column_step(cfg, u, prev, mask))
        self._states = states

        errors = np.stack([s.e for s in states])
        xs = np.stack([s.x for s in states])

        # Точное сравнение с нулём корректно: нулевая ошибка = точный ноль.
        # Маскированные координаты дают ровно 0 → не считаются активностью.
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

    @property
    def n_columns(self) -> int:
        """Число колонок в ансамбле (для расчёта размерности precision)."""
        return len(self._columns)

    @property
    def column_configs(self) -> list[ColumnConfig]:
        """Конфигурации колонок (read-only копия, порядок = индексы аттрактора)."""
        return list(self._columns)

    @property
    def resolved_reads(self) -> tuple[tuple[str, ...], ...]:
        """Фактические reads колонок (наблюдаемость, ADR-0013 §8).

        Кортеж по колонкам; каждая — имена разрешённых сегментов в порядке
        карты. ``reads=None`` → все сегменты карты (``m ≡ 1``). Без
        ``segment_dims`` (проекции не задействованы) → пустые кортежи.
        """
        if not self._segment_dims:
            return tuple(() for _ in self._columns)
        return tuple(
            segment_names(mask, self._segment_dims)
            if mask is not None
            else tuple(self._segment_dims)
            for mask in self._masks
        )
