"""Imperative Shell — TaskAttractor stateful manager.

Functional Core / Imperative Shell (ADR-0004):
- Core (compute.py) — чистые функции, тестируются без Shell
- Shell (TaskAttractor) — хранит состояния, управляет переключениями
"""

from __future__ import annotations

import logging

import numpy as np

from src.core.attractors.compute import (
    check_basin_stability,
    check_immediate_switch,
    compute_dwell,
)
from src.core.attractors.models import TaskAttraction, Vector

logger = logging.getLogger(__name__)


class TaskAttractor:
    """Управляет динамикой аттрактора задачи.

    Functional Core / Imperative Shell (ADR-0004):
    - Core (compute_dwell, check_basin_stability, check_immediate_switch) —
      чистые функции, тестируются без Shell
    - Shell (TaskAttractor) — хранит состояния (mask, history), управляет
      переключениями

    Механизм переключения: SHORT-TERM PLASTICITY (STP)
    по Kubota & Aihara (2011). Не attractor-state itinerancy (стохастический
    переход по фиксированному ландшафту), а детерминированная реконфигурация:
    текущий аттрактор динамически усиливается при удержании и ослабевает
    при конкуренции. Это меняет сам ландшафт ям, а не просто толкает
    систему между фиксированными ямами.

    Ссылка: Kubota & Aihara, "Neural network model of short-term
    plasticity for working memory and attractor dynamics", 2011.
    """

    def __init__(
        self,
        n_tasks: int,
        base_dwell: int = 5,
        dwell_slope: float = 2.0,
        plasticity_gain: float = 0.1,
        basin_threshold: float = 0.15,
        convergence_threshold: float = 1e-8,
    ) -> None:
        """Инициализация аттрактора.

        Args:
            n_tasks: N — число колонок в ансамбле (размер mask).
            base_dwell: α — базовое время удержания в тиках.
                Минимальное время, которое аттрактор удерживается
                до первого разрешённого переключения.
            dwell_slope: β — чувствительность dwell time к разнице scores.
                Формула: min_dwell = α + β · (score_current - score_runner_up).
                При выигрыше (score_current > runner_up) dwell растёт (гистерезис).
                При проигрыше — падает (но не ниже base_dwell).
            plasticity_gain: γ — прирост устойчивости за каждый тик удержания.
                Реализует STP (кратковременную пластичность): чем дольше
                удерживаем задачу, тем сильнее становимся (реконфигурация ландшафта).
            basin_threshold: ε — порог для basin of attraction.
                Если score_runner_up - score_current < ε, считаем,
                что мы в бассейне текущего аттрактора (устойчив к локальному шуму).
            convergence_threshold: τ — порог сходимости EMA.
                Если max(scores) < τ, все scores ~0 (система сходится).
                При converged=True: переключение запрещено, пока
                history_size < min_dwell (защита от флуктуаций нуля).

        Raises:
            ValueError: Если n_tasks < 2 (не существует runner-up).
        """
        if n_tasks < 2:
            raise ValueError(
                f"n_tasks < 2 ({n_tasks}): не существует runner-up для сравнения"
            )

        self._n_tasks: int = n_tasks
        self._base_dwell: int = base_dwell
        self._dwell_slope: float = dwell_slope
        self._plasticity_gain: float = plasticity_gain
        self._basin_threshold: float = basin_threshold
        self._convergence_threshold: float = convergence_threshold

        # Состояние
        self._mask: Vector | None = None
        self._history_size: int = 0

    def tick(self, scores: Vector) -> TaskAttraction:
        """Один тик: обновить аттрактор на основе полного вектора активностей.

        Алгоритм:
        0. Первый тик (self._mask is None):
           - mask = one-hot для argmax(scores), history_size = 0
           - converged = max(scores) < convergence_threshold
           - Return TaskAttraction(mask, history_size, scores.copy(), converged)
        1. Определить held_index — индекс победителя текущей маски (1.0 в mask).
        2. Найти score_runner_up = max(scores[i] for i in range(n_tasks)
           if i != held_index).
        3. Найти score_current = scores[held_index].
        4. Проверить immediate_switch:
           - Если check_immediate_switch(score_current, score_runner_up):
             switch (mask = one-hot для argmax(scores), history_size = 0).
        5. Иначе (нет явного превосходства конкурента):
           a. Вычислить min_dwell = compute_dwell(...).
           b. Если history_size < min_dwell: stay (history += 1).
           c. Иначе (history_size >= min_dwell):
              - switch_now = (argmax(scores) != held_index)
                            and not check_basin_stability(...)
              - Если switch_now: switch (mask = one-hot, history_size = 0)
              - Иначе: stay (history_size += 1).
        6. converged = max(scores) < convergence_threshold.
        7. Return TaskAttraction(mask, history_size, scores.copy(), converged).

        Args:
            scores: Полный вектор активностей всех N колонок (‖e‖²).

        Returns:
            TaskAttraction — снимок аттрактора текущего тика.
        """
        # Шаг 0: первый тик
        if self._mask is None:
            mask = np.zeros(self._n_tasks, dtype=np.float64)
            winner_idx = int(np.argmax(scores))
            mask[winner_idx] = 1.0
            converged = bool(np.max(scores) < self._convergence_threshold)
            result = TaskAttraction(
                mask=mask,
                history_size=0,
                scores=scores.copy(),
                converged=converged,
            )
            self._mask = mask
            self._history_size = 0
            return result

        # Шаг 1: найти held_index
        held_indices = np.where(self._mask == 1.0)[0]
        held_index = held_indices[0]
        score_current = float(scores[held_index])

        # Шаг 2: найти score_runner_up
        score_runner_up = max(
            float(scores[i]) for i in range(self._n_tasks) if i != held_index
        )

        # Шаг 4: немедленное переключение
        if check_immediate_switch(score_current, score_runner_up):
            mask = np.zeros(self._n_tasks, dtype=np.float64)
            winner_idx = int(np.argmax(scores))
            mask[winner_idx] = 1.0
            self._mask = mask
            self._history_size = 0
            converged = bool(np.max(scores) < self._convergence_threshold)
            logger.debug(
                "attractor: immediate_switch held=%d winner=%d "
                "Δ=%.4f",
                held_index, winner_idx, score_runner_up - score_current,
            )
            return TaskAttraction(
                mask=mask,
                history_size=0,
                scores=scores.copy(),
                converged=converged,
            )

        # Шаг 5: нет явного превосходства
        min_dwell = compute_dwell(
            self._base_dwell,
            self._dwell_slope,
            self._plasticity_gain,
            score_current,
            score_runner_up,
            self._history_size,
        )

        if self._history_size < min_dwell:
            # Шаг 5b: cool-down, stay
            self._history_size += 1
            converged = bool(np.max(scores) < self._convergence_threshold)
            return TaskAttraction(
                mask=self._mask.copy(),
                history_size=self._history_size,
                scores=scores.copy(),
                converged=converged,
            )

        # Шаг 5c: history_size >= min_dwell, проверяем basin
        current_winner_idx = int(np.argmax(scores))
        switch_now = (current_winner_idx != held_index) and not check_basin_stability(
            self._basin_threshold,
            score_current,
            score_runner_up,
            self._history_size,
        )

        if switch_now:
            mask = np.zeros(self._n_tasks, dtype=np.float64)
            mask[current_winner_idx] = 1.0
            self._mask = mask
            self._history_size = 0
            logger.debug(
                "attractor: switch basin held=%d winner=%d "
                "Δ=%.4f",
                held_index, current_winner_idx,
                score_runner_up - score_current,
            )
        else:
            # Stay: либо held и так топ-1, либо в бассейне
            self._history_size += 1

        converged = bool(np.max(scores) < self._convergence_threshold)
        return TaskAttraction(
            mask=self._mask.copy(),
            history_size=self._history_size,
            scores=scores.copy(),
            converged=converged,
        )

    def reset(self) -> None:
        """Сбросить аттрактор: mask = None, history = 0.

        Следующий tick() попадёт в ветку «первый тик» и заново выберет
        победителя.
        """
        self._mask = None
        self._history_size = 0

    @property
    def current_mask(self) -> Vector | None:
        """Текущий аттрактор (mask). None до первого tick()."""
        if self._mask is None:
            return None
        return self._mask.copy()

    @property
    def history_size(self) -> int:
        """Сколько тиков подряд аттрактор не менялся."""
        return self._history_size
