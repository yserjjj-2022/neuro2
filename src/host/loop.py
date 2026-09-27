"""Host loop — the beating heart of the host.

Each tick (S2, memory wired in):
    dt, now = time source (synthetic: tick·tick_dt; wall: measured)
    u_base  = SignalBus.step(tick, now)
    text    = message_provider.text_at(tick)     (S2)
    query   = memory.context_embedding(text)      (S2, cached)
    prior   = memory.recall_prior(query)          (S2)
    u       = concat(u_base, prior)               (S2)
    γ       = PrecisionEstimator.update(u)  (or ones baseline)
    outcome = pipeline.tick(u, γ, dt, segments, reflex_tags)
    guard   = check_finite(outcome.result)
    drift   = DriftDetector.update(outcome.result)
    stored  = memory.maybe_store(...)             (S2)
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
from src.host.sources import BusSegment, SignalBus, default_providers
from src.host.text_source import TextMessageProvider
from src.host.wiring import CMCPipeline, TickOutcome, build_cmc_pipeline
from src.memory import (
    MemoryRouter,
    MemoryStore,
    build_embedder,
    embedder_settings_from_env,
)
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
        time_scale: Множитель субъективного времени.
        memory: Роутер памяти (S2); None → контур без памяти (S1).
        message_provider: Коммуникативный вход (S2); None → нет текста.
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
    memory: MemoryRouter | None = None
    message_provider: TextMessageProvider | None = None
    clock: Callable[[], float] = time.time
    _prev_now: float | None = field(default=None, init=False, repr=False)
    _prev_f: float = field(default=0.0, init=False, repr=False)
    _last_text: str = field(default="", init=False, repr=False)
    _cached_query: np.ndarray | None = field(default=None, init=False, repr=False)
    _cached_prior: np.ndarray | None = field(default=None, init=False, repr=False)
    last_outcome: TickOutcome | None = field(default=None, init=False, repr=False)
    _spoke_pending: bool = field(default=False, init=False, repr=False)

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
        # Память включена, но сообщений ещё не было → нулевой приор
        # (ширина шины стабильна: prior всегда присутствует).
        if self.memory is not None and self._cached_prior is None:
            self._cached_prior = np.zeros(self.memory.prior_dim, dtype=np.float64)

    @property
    def total_dim(self) -> int:
        """Полная входная размерность колонок: шина + приор памяти."""
        prior_dim = self.memory.prior_dim if self.memory is not None else 0
        return self.bus.bus_dim + prior_dim

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
            u: Вектор шины (с приором памяти) shape=(total_dim,).

        Returns:
            Вектор γ shape=(N_columns * total_dim,).
        """
        n_columns = self.pipeline.ensemble.n_columns
        if self.precision_mode == "ones":
            gamma_bus = np.ones(u.shape[0], dtype=np.float64)
        else:
            gamma_bus = self.estimator.update(u)
        return np.tile(gamma_bus, n_columns)

    def _update_prior_if_new_text(self, text: str) -> np.ndarray | None:
        """Обновить приор памяти только при новом тексте (event-triggered).

        Эмбеддинг и recall — дорогие (сеть, SQLite); текст приходит редко,
        а тики частые. При неизменном тексте используется сохранённый приор.

        Args:
            text: Текущий текст собеседника ("" → нет сообщения).

        Returns:
            Эмбеддинг нового текста (query) или None; приор доступен через
            ``_cached_prior``.
        """
        if self.memory is None:
            return None
        if text == self._last_text:
            return self._cached_query

        self._last_text = text
        if not text:
            self._cached_query = None
            self._cached_prior = np.zeros(self.memory.prior_dim, dtype=np.float64)
            return None

        query = self.memory.context_embedding(text)
        self._cached_query = query
        self._cached_prior = self.memory.recall_prior(query)
        return query

    def step_once(self, tick: int) -> TickOutcome:
        """Один тик: время → шина → память → precision → pipeline → guard → log.

        Коммуникативный вход — **событийный**: эмбеддинг и recall запускаются
        только при появлении нового текста (сравнение с предыдущим). Между
        сообщениями используется сохранённый приор — сеть не трогается на
        каждом тике (манифест §3.Е, ADR-0006: непрерывный аффективный контур,
        event-triggered рациональный).

        Args:
            tick: Номер тика (влияет на шину, провайдеры и время).

        Returns:
            TickOutcome текущего тика.

        Raises:
            HostIntegrityError: Если аффективные метрики не конечны.
        """
        dt, now = self._time_for_tick(tick)

        # Сенсорная фаза (включая эмбеддинг — сетевой I/O) НЕ входит в
        # ресурсную латентность: сеть — внешняя нагрузка, а «тахикардия»
        # измеряет собственные вычисления хоста.
        u_base = self.bus.step(tick, now)
        text = self.message_provider.text_at(tick) if self.message_provider else ""
        has_new_message = bool(text) and text != self._last_text
        query = self._update_prior_if_new_text(text)
        prior = self._cached_prior
        u = np.concatenate([u_base, prior]) if prior is not None else u_base

        compute_start = time.perf_counter()
        gamma = self.precision(u)
        segments = self.bus.segments
        if self.memory is not None:
            segments = segments + (
                BusSegment(
                    name="memory",
                    offset=self.bus.bus_dim,
                    dim=self.memory.prior_dim,
                    period=1,
                ),
            )
        reflex_tags = tuple(s.tag for s in self.bus.last_signals if s.is_reflex)
        outcome = self.pipeline.tick(u, gamma, dt, segments, reflex_tags)
        check_finite(outcome.result)
        drift = self.drift.update(outcome.result)
        self.meter.record_tick(time.perf_counter() - compute_start)

        memory_prior_value = 0.0
        memory_hit = False
        episode_stored = False
        if self.memory is not None and prior is not None:
            memory_prior_value = float(prior[0])
            memory_hit = bool(np.any(prior != 0.0))
            stored_id = self.memory.maybe_store(
                text=text,
                query=query,
                f=outcome.result.f,
                prev_f=self._prev_f,
                valence=outcome.result.valence,
                stress=outcome.result.allostatic_stress,
                active_tags=outcome.active_tags,
                reflex_tags=outcome.reflex_tags,
                now=now,
                has_new_message=has_new_message,
            )
            episode_stored = stored_id is not None

        self._prev_f = outcome.result.f

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
            memory_prior=memory_prior_value,
            memory_hit=memory_hit,
            episode_stored=episode_stored,
            spoke=self._spoke_pending,
        )
        self._spoke_pending = False
        self.last_outcome = outcome
        return outcome

    def mark_spoke(self) -> None:
        """Отметить, что хост сгенерировал реплику (попадёт в телеметрию)."""
        self._spoke_pending = True

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
        """Graceful shutdown: закрыть телеметрию и store (идемпотентно)."""
        writer = self.logger.writer
        if hasattr(writer, "close"):
            writer.close()
        if self.memory is not None:
            store = self.memory.store
            if hasattr(store, "close"):
                store.close()


def build_host_loop(
    config: HostConfig | None = None,
    meter: ResourceMeter | None = None,
    messages: tuple[tuple[int, str], ...] = (),
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
        messages: Скрипт коммуникативных сообщений ``(tick, text)`` для
            ``TextMessageProvider`` (S2; в S3 заменится живым вводом).

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

    memory: MemoryRouter | None = None
    message_provider: TextMessageProvider | None = None
    if config.memory.enabled:
        api = embedder_settings_from_env()
        embedder = build_embedder(
            mode=config.memory.embedder_mode,
            dim=config.memory.embedding_dim,
            model=str(api["model"]),
            base_url=str(api["base_url"]),
            api_dim=int(api["api_dim"]),  # type: ignore[arg-type]
        )
        store = MemoryStore(db_path=config.memory.db_path, embedding_dim=embedder.dim)
        memory = MemoryRouter(
            store=store,
            embedder=embedder,
            spike_threshold=config.memory.episode_spike_threshold,
            recall_limit=config.memory.recall_limit,
            prior_dim=config.memory.prior_dim,
        )
        message_provider = TextMessageProvider(embedder=embedder, messages=messages)

    providers = default_providers(
        message_dim=config.memory.embedding_dim,
        seed=config.seed,
        resource_provider=resource_provider,
        message_provider=message_provider,
    )
    bus = SignalBus(providers)
    prior_dim = config.memory.prior_dim if memory is not None else 0
    total_dim = bus.bus_dim + prior_dim

    columns = [
        params.build(input_dim=total_dim, state_dim=total_dim)
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
            dim=total_dim,
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
        memory=memory,
        message_provider=message_provider,
    )
