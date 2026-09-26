"""Host loop — the beating heart of the host.

Each tick:
    u(t) = SignalBus.step(tick, now)          # wide sensory bus
    precision = ones(N_columns * bus_dim)     # Phase 1 baseline (γ = 1)
    pipeline.tick(u, precision)               # cmc → voting/attractors → energy
                                              # → telemetry (JSONL)

The loop owns the wall clock and passes ``now`` to the bus, keeping providers
pure. ``dt`` is the integration step: the loop sleeps ``dt`` between ticks
(``dt = 0`` → free-run, useful for tests and batch runs).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import HostConfig
from src.core.energy import FreeEnergyResult
from src.host.sources import SignalBus, default_providers
from src.host.wiring import CMCPipeline, build_cmc_pipeline


@dataclass
class HostLoop:
    """Imperative Shell: гоняет полный конвейер тик за тиком.

    Attributes:
        bus: Сенсорная шина (провайдеры → ``u(t)``).
        pipeline: Полный per-tick конвейер.
        dt: Шаг интегрирования в секундах (0 → без пауз).
        clock: Источник wall-clock времени (инъекция для тестов).
    """

    bus: SignalBus
    pipeline: CMCPipeline
    dt: float = 0.01
    precision_mode: str = "ones"
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        """Валидация: dt не может быть отрицательным, режим — известным."""
        if self.dt < 0.0:
            raise ValueError(f"dt must be >= 0, got {self.dt}")
        if self.precision_mode not in ("ones", "variance"):
            raise ValueError(
                f"precision_mode must be 'ones' or 'variance', "
                f"got {self.precision_mode!r}"
            )

    def precision(self) -> np.ndarray:
        """Точность γ для всех каналов.

        Фаза 1 — baseline ``ones`` (γ = 1). Режим ``variance`` (γ = 1/var)
        зарезервирован: реализация оконной дисперсии — Фаза 2
        (BACKLOG `[Phase2][energy] Оконная дисперсия precision`).

        Returns:
            Вектор γ shape=(N_columns * bus_dim,).

        Raises:
            NotImplementedError: Если выбран режим ``variance`` (Фаза 2).
        """
        if self.precision_mode == "variance":
            raise NotImplementedError(
                "precision_mode='variance' is Phase 2 (BACKLOG [Phase2][energy])"
            )
        n_columns = self.pipeline.ensemble.n_columns
        return np.ones(n_columns * self.bus.bus_dim, dtype=np.float64)

    def run(self, max_ticks: int) -> int:
        """Прогнать цикл: до ``max_ticks`` тиков.

        Args:
            max_ticks: Максимальное число тиков (0 → ничего не делать).

        Returns:
            Фактически выполненные тики.

        Raises:
            ValueError: Если max_ticks < 0.
        """
        if max_ticks < 0:
            raise ValueError(f"max_ticks must be >= 0, got {max_ticks}")

        precision = self.precision()
        for tick in range(max_ticks):
            now = self.clock()
            u = self.bus.step(tick, now)
            self.pipeline.tick(u, precision)
            if self.dt > 0.0 and tick < max_ticks - 1:
                time.sleep(self.dt)

        return max_ticks

    def step_once(self, tick: int) -> FreeEnergyResult:
        """Один тик без пауз — для тестов и ручного управления.

        Args:
            tick: Номер тика (влияет на шину и провайдеры).

        Returns:
            FreeEnergyResult текущего тика.
        """
        now = self.clock()
        u = self.bus.step(tick, now)
        return self.pipeline.tick(u, self.precision())

    def close(self) -> None:
        """Graceful shutdown: закрыть телеметрию."""
        self.pipeline.close()


def build_host_loop(config: HostConfig | None = None) -> HostLoop:
    """Собрать host loop: провайдеры → шина → колонки → конвейер.

    Ширина шины определяется составом провайдеров, поэтому колонки создаются
    под фактический ``bus_dim`` — конфигурация не разъезжается. Все пороги
    берутся из ``HostConfig`` (CONSTITUTION §2.2), не хардкодятся здесь.

    Args:
        config: Полная конфигурация хоста. None → HostConfig() с дефолтами.

    Returns:
        Готовый к ``run()`` HostLoop.

    Raises:
        ValueError: Если число колонок меньше k (ансамбль не соберётся).
    """
    if config is None:
        config = HostConfig()

    providers = default_providers(message_dim=config.message_dim, seed=config.seed)
    bus = SignalBus(providers)

    columns = [
        params.build(input_dim=bus.bus_dim, state_dim=bus.bus_dim)
        for params in config.columns
    ]
    if len(columns) < config.k:
        raise ValueError(f"k={config.k} exceeds number of columns {len(columns)}")

    pipeline = build_cmc_pipeline(
        columns=columns,
        k=config.k,
        log_path=Path(config.log_path),
        active_threshold=config.active_threshold,
        attractor=config.attractor.build(n_tasks=len(columns)),
        calculator=config.energy.build(),
    )
    return HostLoop(
        bus=bus,
        pipeline=pipeline,
        dt=config.dt,
        precision_mode=config.precision_mode,
    )
