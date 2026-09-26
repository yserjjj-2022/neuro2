"""Host loop — the beating heart of the host.

Each tick (S1, single time base):
    dt, now = time source (synthetic: tick·tick_dt; wall: measured)
    u(t)    = SignalBus.step(tick, now)
    γ       = PrecisionEstimator.update(u)  (or ones baseline)
    outcome = pipeline.tick(u, γ, dt, segments, reflex_tags)
    guard   = check_finite(outcome.result)
    drift   = DriftDetector.update(outcome.result)
    telemetry.log(...)  (loop owns the writer — it has the full context)

The loop owns the clock and telemetry. ``clock_mode="synthetic"`` (default)
gives deterministic, replay-friendly runs; ``"wall"`` measures real time.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import HostConfig
from src.core.energy import DriftDetector, PrecisionEstimator, check_finite
from src.host.resources import ResourceMeter, ResourceProvider
from src.host.sources import SignalBus, default_providers
from src.host.wiring import CMCPipeline, TickOutcome, build_cmc_pipeline
from src.telemetry import TelemetryLogger, TelemetryWriter


@dataclass
class HostLoop:
    """Imperative Shell: гоняет полный конвейер тик за тиком.

    Attributes:
        bus: Сенсорная шина (провайдеры → ``u(t)``).
        pipeline: Per-tick конвейер (без I/O).
        logger: Телеметрия (loop владеет writer'ом).
        estimator: Оценщик точности γ.
        meter: Ресурсный meter (латентность/RSS).
        drift: Детектор дрейфа (заготовка).
        tick_dt: Номинальная длительность тика, с (> 0).
        clock_mode: "synthetic" (tick·tick_dt) или "wall" (реальное время).
        paced: Спать между тиками, чтобы реальное время ≈ tick_dt.
        precision_mode: "variance" (γ=1/var) или "ones" (baseline).
        clock: Источник wall-clock (инъекция для тестов).
    """

    bus: SignalBus
    pipeline: CMCPipeline
    logger: TelemetryLogger
    estimator: PrecisionEstimator
    meter: ResourceMeter
    drift: DriftDetector
    tick_dt: float = 0.01
    clock_mode: str = "synthetic"
    paced: bool = False
    precision_mode: str = "variance"
    time_scale: float = 1.0
    clock: Callable[[], float] = time.time
    _prev_now: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tick_dt <= 0.0:
            raise ValueError(f"tick_dt must be > 0, got {self.tick_dt}")
        if self.clock_mode not in ("synthetic", "wall"):
            raise ValueError(
                f"clock_mode must be 'synthetic' or 'wall', got {self.clock_mode!r}"
            )
        if self.precision_mode not in ("variance", "ones"):
            raise ValueError(
                f"precision_mode must be 'variance' or 'ones', "
                f"got {self.precision_mode!r}"
            )
        if self.time_scale <= 0.0:
            raise ValueError(f"time_scale must be > 0, got {self.time_scale}")

    def _time_for_tick(self, tick: int) -> tuple[float, float]:
        """Вернуть (dt, now) для тика в соответствии с clock_mode и time_scale.

        ``time_scale`` масштабирует субъективное время: 1.0 = жизнь,
        >1 = ускоренная симуляция (ADR-0006).

        Args:
            tick: Номер тика (с 0).

        Returns:
            (dt, now) — шаг интегрирования и wall-clock время, секунды.
        """
        if self.clock_mode == "synthetic":
            dt = self.tick_dt * self.time_scale
            return dt, tick * dt

        now = self.clock()
        if self._prev_now is None:
            dt = self.tick_dt
        else:
            dt = max(now - self._prev_now, 1e-9)
        self._prev_now = now
        return dt, now

    def precision(self, u: np.ndarray) -> np.ndarray:
        """Точность γ для всех каналов.

        ``variance`` — γ = 1/var по окну (PrecisionEstimator).
        ``ones`` — baseline (фоллбэк FEP, ADR-0005 §4).

        Args:
            u: Вектор шины shape=(bus_dim,).

        Returns:
            Вектор γ shape=(N_columns * bus_dim,).
        """
        n_columns = self.pipeline.ensemble.n_columns
        if self.precision_mode == "ones":
            gamma_bus = np.ones(self.bus.bus_dim, dtype=np.float64)
        else:
            gamma_bus = self.estimator.update(u)
        return np.tile(gamma_bus, n_columns)

    def step_once(self, tick: int) -> TickOutcome:
        """Один тик: время → шина → precision → pipeline → guard → drift → log.

        Args:
            tick: Номер тика (влияет на шину, провайдеры и время).

        Returns:
            TickOutcome текущего тика.

        Raises:
            HostIntegrityError: Если аффективные метрики не конечны.
        """
        dt, now = self._time_for_tick(tick)

        start = time.perf_counter()
        u = self.bus.step(tick, now)
        gamma = self.precision(u)
        reflex_tags = tuple(s.tag for s in self.bus.last_signals if s.is_reflex)
        outcome = self.pipeline.tick(u, gamma, dt, self.bus.segments, reflex_tags)
        check_finite(outcome.result)
        drift = self.drift.update(outcome.result)
        self.meter.record_tick(time.perf_counter() - start)

        self.logger.log(
            free_energy=outcome.result.f,
            valence=outcome.result.valence,
            allostatic_stress=outcome.result.allostatic_stress,
            active_columns=self.pipeline.ensemble.active,
            tick=tick,
            gamma=outcome.result.gamma,
            active_tags=",".join(outcome.active_tags),
            reflex_tags=",".join(outcome.reflex_tags),
            bus_dim=self.bus.bus_dim,
            latency_ms=self.meter.last_latency_s * 1000.0,
            rss_mb=self.meter.last_rss_mb,
            drift=drift,
        )
        return outcome

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

        for tick in range(max_ticks):
            self.step_once(tick)
            if self.paced and tick < max_ticks - 1:
                time.sleep(self.tick_dt)
        return max_ticks

    def close(self) -> None:
        """Graceful shutdown: закрыть телеметрию (идемпотентно)."""
        writer = self.logger.writer
        if hasattr(writer, "close"):
            writer.close()


def build_host_loop(
    config: HostConfig | None = None,
    meter: ResourceMeter | None = None,
) -> HostLoop:
    """Собрать host loop: провайдеры → шина → колонки → конвейер → телеметрия.

    Ширина шины определяется составом провайдеров, поэтому колонки создаются
    под фактический ``bus_dim``. Все пороги берутся из ``HostConfig``.

    Args:
        config: Полная конфигурация хоста. None → HostConfig() с дефолтами.
        meter: Источник ресурсных метрик (инъекция). None → реальный
            ResourceMeter. Для детерминированного replay/synthetic-прогонов
            следует передать fake-meter (реальный RSS различается между
            запусками — см. ADR-0006, stages/S1_SPEC.md §5).

    Returns:
        Готовый к ``run()`` HostLoop.

    Raises:
        ValueError: Если число колонок меньше k (ансамбль не соберётся).
    """
    if config is None:
        config = HostConfig()

    if meter is None:
        meter = ResourceMeter()
    resource_provider = ResourceProvider(
        meter=meter,
        tick_budget_ms=config.tick_budget_ms,
        rss_budget_mb=config.rss_budget_mb,
    )
    providers = default_providers(
        message_dim=config.message_dim,
        seed=config.seed,
        resource_provider=resource_provider,
    )
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

    writer = TelemetryWriter(log_path=Path(config.log_path))
    logger = TelemetryLogger(writer=writer, phase="phase1", mode="free")

    return HostLoop(
        bus=bus,
        pipeline=pipeline,
        logger=logger,
        estimator=PrecisionEstimator(
            dim=bus.bus_dim,
            window=config.precision_window,
            eps=config.precision_eps,
            gamma_max=config.gamma_max,
        ),
        meter=meter,
        drift=DriftDetector(
            f_threshold=config.drift_f_threshold,
            stress_threshold=config.drift_stress_threshold,
        ),
        tick_dt=config.dt,
        clock_mode=config.clock_mode,
        paced=config.paced,
        precision_mode=config.precision_mode,
        time_scale=config.time_scale,
    )
