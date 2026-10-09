"""Sensitivity harness — knob perturbation matrix + direction invariants (S7-A).

Functional Core (pure):

* :class:`SensitivityCase` — one matrix cell: a knob, the values to sweep, the
  fingerprint metric to watch and the expected direction.
* :func:`build_sensitivity_matrix` — hardcodes *which* knobs and *which ranges*
  are perturbed (ADR-0010 §3). The matrix checks **directions**, not exact
  values: exact values are calibration (BACKLOG ``[S4][policy]``).
* :func:`check_direction` — verifies an ordering/boundedness invariant.

Imperative Shell:

* :class:`SensitivityRunner` — runs the host loop once per knob value, collects
  a :class:`BehavioralFingerprint` and extracts the named metric. Runs are
  deterministic (synthetic clock, fake embedder, fake resource meter, fixed
  seed), so the same seed yields the same fingerprints.

The harness drives policy itself (no LLM, ADR-0007) so that the speech/initiative
rates are populated: after each tick it evaluates ``select_action`` on the loop's
context and records the trace, mirroring ``ChatSession``.
"""

from __future__ import annotations

import itertools
import json
import math
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.config import AutonomyConfig, HostConfig, MemoryConfig, PolicyConfig
from src.core.policy import Action, Preferences, select_action
from src.host.fingerprint import (
    FINGERPRINT_METRICS,
    BehavioralFingerprint,
    behavioral_fingerprint,
)
from src.host.loop import build_host_loop
from src.host.resources import ResourceMeter

_DIRECTIONS = ("nondecreasing", "nonincreasing", "bounded")
_SPEAKING = frozenset(
    {Action.RESPOND, Action.INITIATIVE, Action.IDENTIFY_PARTNER}
)
# Knobs that live in Preferences (policy).
_PREFERENCE_KNOBS = (
    "silent_stress_gain",
    "initiative_f_threshold",
    "alert_deviation",
    "explore_threshold",
    "silent_baseline",
)


@dataclass(frozen=True)
class SensitivityCase:
    """Одна ячейка матрицы чувствительности.

    Attributes:
        knob: Имя возмущаемой ручки (поле ``Preferences`` или ``SpeechConfig``).
        values: Значения ручки (свип).
        metric: Метрика отпечатка (``talk_rate``, ``explore_rate``, ...).
        direction: Ожидаемый инвариант — ``nondecreasing`` | ``nonincreasing`` |
            ``bounded``.

    Raises:
        ValueError: Если ``values`` пуст, направление/метрика неизвестны.
    """

    knob: str
    values: tuple[float, ...]
    metric: str
    direction: str

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("values must not be empty")
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"direction must be one of {_DIRECTIONS}, got {self.direction!r}"
            )
        if self.metric not in FINGERPRINT_METRICS:
            raise ValueError(
                f"metric must be one of {FINGERPRINT_METRICS}, got {self.metric!r}"
            )


@dataclass(frozen=True)
class SensitivityResult:
    """Итог прогона одной ячейки матрицы.

    Attributes:
        case: Проверяемая ячейка.
        metric_values: Значения метрики по свипу ручки.
        passed: Выполнен ли инвариант направления.
    """

    case: SensitivityCase
    metric_values: tuple[float, ...]
    passed: bool


def build_sensitivity_matrix() -> tuple[SensitivityCase, ...]:
    """Хардкод матрицы возмущений (ручки/диапазоны/направления).

    Возвращает:
        Кортеж :class:`SensitivityCase`. Ручки и диапазоны захардкожены;
        проверяются только инварианты направления/границ (ADR-0010 §3).
    """
    return (
        # Выше ценность покоя со стрессом → реже говорим.
        SensitivityCase(
            knob="silent_stress_gain",
            values=(0.0, 0.5, 2.0),
            metric="talk_rate",
            direction="nonincreasing",
        ),
        # Выше порог F для инициативы → реже инициатива.
        SensitivityCase(
            knob="initiative_f_threshold",
            values=(0.5, 1.0, 5.0),
            metric="initiative_rate",
            direction="nonincreasing",
        ),
        # Выше порог гомеостатической тревоги → реже инициатива.
        SensitivityCase(
            knob="alert_deviation",
            values=(0.2, 0.7, 0.95),
            metric="initiative_rate",
            direction="nonincreasing",
        ),
        # Выше порог неопределённости для EXPLORE → реже исследование.
        SensitivityCase(
            knob="explore_threshold",
            values=(0.2, 0.6, 0.9),
            metric="explore_rate",
            direction="nonincreasing",
        ),
        # Доля реплик остаётся в [0, 1] при любом возмущении.
        SensitivityCase(
            knob="silent_stress_gain",
            values=(0.0, 1.0, 3.0),
            metric="talk_rate",
            direction="bounded",
        ),
    )


def check_direction(
    metric_values: Sequence[float],
    *,
    direction: str,
    tol: float = 1e-9,
) -> bool:
    """Проверить инвариант направления/границ метрики (чистая).

    Args:
        metric_values: Значения метрики в порядке роста ручки.
        direction: ``nondecreasing`` | ``nonincreasing`` | ``bounded``.
        tol: Допуск на монотонность (шум float).

    Returns:
        True, если инвариант выполнен.

    Raises:
        ValueError: Если ``direction`` неизвестен или ``metric_values`` пуст.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}")
    if not metric_values:
        raise ValueError("metric_values must not be empty")
    if not all(math.isfinite(v) for v in metric_values):
        return False
    if direction == "nondecreasing":
        return all(
            b >= a - tol for a, b in itertools.pairwise(metric_values)
        )
    if direction == "nonincreasing":
        return all(
            b <= a + tol for a, b in itertools.pairwise(metric_values)
        )
    return all(0.0 <= v <= 1.0 for v in metric_values)


class DeterministicMeter(ResourceMeter):
    """Детерминированный ресурсный meter для воспроизводимых прогонов."""

    def __init__(self, latency_s: float = 0.0005, rss_mb: float = 100.0) -> None:
        super().__init__()
        self._fake_latency = latency_s
        self._fake_rss = rss_mb

    @property
    def last_latency_s(self) -> float:
        """Фиксированная латентность."""
        return self._fake_latency

    @property
    def last_rss_mb(self) -> float:
        """Фиксированный RSS."""
        return self._fake_rss

    def record_tick(self, latency_s: float) -> None:
        """Игнорировать реальное время (детерминизм)."""


def _base_config(run_dir: Path, *, seed: int, ticks: int) -> HostConfig:
    """Детерминированная база прогона: synthetic, fake embedder, policy, autonomy."""
    return HostConfig(
        dt=0.1,
        max_ticks=ticks,
        seed=seed,
        precision_mode="ones",
        clock_mode="synthetic",
        log_path=str(run_dir / "run.jsonl"),
        memory=MemoryConfig(
            enabled=True, embedder_mode="fake", db_path=str(run_dir / "mem.db")
        ),
        policy=PolicyConfig(enabled=True, mode="free"),
        autonomy=AutonomyConfig(enabled=True),
    )


def _apply_knob(config: HostConfig, knob: str, value: float) -> HostConfig:
    """Вернуть конфиг с возмущённой ручкой (fail-fast для неизвестной ручки).

    Args:
        config: Базовый конфиг.
        knob: Имя ручки.
        value: Новое значение.

    Returns:
        Новый HostConfig.

    Raises:
        ValueError: Если ручка неизвестна.
    """
    if knob in _PREFERENCE_KNOBS:
        updates: dict[str, Any] = {knob: value}
        preferences: Preferences = replace(
            config.policy.preferences, **updates
        )
        return replace(config, policy=replace(config.policy, preferences=preferences))
    if knob == "f_threshold":
        return replace(config, speech=replace(config.speech, f_threshold=value))
    raise ValueError(f"unknown sensitivity knob {knob!r}")


def _read_events(log_path: Path) -> list[dict]:
    """Прочитать JSONL телеметрии в список словарей."""
    return [json.loads(line) for line in log_path.read_text().strip().split("\n")]


class SensitivityRunner:
    """Shell: прогон harness-кейсов поверх HostLoop, сбор отпечатков.

    Attributes:
        ticks: Число тиков на прогон.
        workdir: Каталог для логов/БД (детерминизм не зависит от пути).
    """

    def __init__(self, *, ticks: int = 120, workdir: Path | None = None) -> None:
        if ticks < 1:
            raise ValueError(f"ticks must be >= 1, got {ticks}")
        self.ticks = ticks
        self.workdir = (
            Path(workdir)
            if workdir is not None
            else Path(tempfile.mkdtemp(prefix="sensitivity-"))
        )
        self._counter = 0

    def run_case(
        self, case: SensitivityCase, *, seed: int, ticks: int | None = None
    ) -> tuple[float, ...]:
        """Прогнать свип ручки и вернуть значения метрики.

        Args:
            case: Ячейка матрицы.
            seed: Зерно детерминированных провайдеров.
            ticks: Тиков на прогон (None → ``self.ticks``).

        Returns:
            Значения ``case.metric`` в порядке ``case.values``.
        """
        n = self.ticks if ticks is None else ticks
        return tuple(
            self._run_value(case, value, seed=seed, ticks=n) for value in case.values
        )

    def run_all(
        self, *, seed: int, ticks: int | None = None
    ) -> list[SensitivityResult]:
        """Прогнать всю матрицу и вернуть результаты с вердиктами.

        Args:
            seed: Зерно.
            ticks: Тиков на прогон.

        Returns:
            Список :class:`SensitivityResult` по каждой ячейке.
        """
        results = []
        for case in build_sensitivity_matrix():
            values = self.run_case(case, seed=seed, ticks=ticks)
            passed = check_direction(values, direction=case.direction)
            results.append(SensitivityResult(case, values, passed))
        return results

    def _run_value(
        self, case: SensitivityCase, value: float, *, seed: int, ticks: int
    ) -> float:
        """Прогнать loop при одном значении ручки и вернуть метрику."""
        self._counter += 1
        run_dir = self.workdir / f"run{self._counter:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _apply_knob(_base_config(run_dir, seed=seed, ticks=ticks), case.knob, value)
        fingerprint = self._run_loop(config, ticks=ticks)
        return fingerprint.metric(case.metric)

    @staticmethod
    def _run_loop(config: HostConfig, *, ticks: int) -> BehavioralFingerprint:
        """Прогнать loop, ведя policy вручную, и вернуть отпечаток."""
        loop = build_host_loop(config, meter=DeterministicMeter())
        try:
            for tick in range(ticks):
                loop.step_once(tick)
                context = loop.policy_context(
                    has_new_message=False, mode=config.policy.mode
                )
                trace = select_action(context, config.policy.preferences)
                loop.record_policy(trace)
                if trace.chosen in _SPEAKING:
                    loop.mark_spoke()
        finally:
            loop.close()
        return behavioral_fingerprint(_read_events(Path(config.log_path)))
