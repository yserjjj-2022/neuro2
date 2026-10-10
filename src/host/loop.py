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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import ActuationConfig, AutonomyConfig, HostConfig, PolicyConfig
from src.core.actuation import Actuation, ActuationKind, Node
from src.core.cmc import apply_attention, attention_gate
from src.core.energy import DriftDetector, PrecisionEstimator, check_finite
from src.core.homeostasis import HomeostasisState, Homeostat
from src.core.policy import MacroContext, PartnerView, PolicyContext, PolicyTrace
from src.core.selfcontrol import SelfMonitor
from src.host.effectors import Effector, SpeechEffector, ToolEffector
from src.host.executor import ActuatorExecutor, ExecutorOutcome
from src.host.gate import (
    Capability,
    CapabilityGate,
    CapabilityTier,
)
from src.host.probe import ProbeEffector, ProbeFn
from src.host.resources import ResourceMeter, ResourceProvider
from src.host.sources import (
    BusSegment,
    SignalBus,
    channel_importance,
    default_providers,
)
from src.host.text_source import TextMessageProvider
from src.host.throttle import ThrottlePlan, plan_throttle
from src.host.wiring import CMCPipeline, TickOutcome, build_cmc_pipeline
from src.integrations import IntegrationRegistry, to_affordances
from src.mcp.probe import (
    AffordanceMap,
    ProbeRequest,
    ProbeResult,
    default_affordances,
    select_affordance,
)
from src.memory import (
    MemoryRouter,
    MemoryStore,
    build_embedder,
    embedder_settings_from_env,
)
from src.memory.consolidation import should_consolidate
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
        importance: Веса важности каналов шины (по компонентам, с учётом
            приора памяти). None → единицы (обратная совместимость).
        clock: Источник wall-clock (инъекция для тестов).
    """

    bus: SignalBus
    pipeline: CMCPipeline
    logger: TelemetryLogger
    estimator: PrecisionEstimator
    meter: ResourceMeter
    drift: DriftDetector
    homeostat: Homeostat | None = None
    throttle_k_scale: float = 0.5
    throttle_dt_scale: float = 2.0
    throttle_severity_threshold: float = 0.9
    escape_hatch_ticks: int = 3
    attention_gate: bool = False
    attention_gamma_ref: float = 1.0
    attention_floor: float = 0.0
    policy_config: PolicyConfig | None = None
    tick_dt: float = 0.01
    clock_mode: str = "synthetic"
    paced: bool = False
    precision_mode: str = "variance"
    time_scale: float = 1.0
    memory: MemoryRouter | None = None
    message_provider: TextMessageProvider | None = None
    recall_enabled: bool = True
    selfcontrol: SelfMonitor | None = None
    autonomy_config: AutonomyConfig | None = None
    probe_effector: ProbeEffector | None = None
    actuation_config: ActuationConfig | None = None
    executor: ActuatorExecutor | None = None
    importance: np.ndarray | None = None
    clock: Callable[[], float] = time.time
    _prev_now: float | None = field(default=None, init=False, repr=False)
    _prev_f: float = field(default=0.0, init=False, repr=False)
    _last_text: str = field(default="", init=False, repr=False)
    _cached_query: np.ndarray | None = field(default=None, init=False, repr=False)
    _cached_prior: np.ndarray | None = field(default=None, init=False, repr=False)
    last_outcome: TickOutcome | None = field(default=None, init=False, repr=False)
    last_drift: bool = field(default=False, init=False, repr=False)
    last_memory_hit: bool = field(default=False, init=False, repr=False)
    last_homeostasis: HomeostasisState | None = field(
        default=None, init=False, repr=False
    )
    last_throttle: ThrottlePlan = field(
        default_factory=lambda: ThrottlePlan(active=False),
        init=False,
        repr=False,
    )
    last_policy_trace: PolicyTrace | None = field(default=None, init=False, repr=False)
    _policy_action_pending: str = field(default="", init=False, repr=False)
    _policy_reason_pending: str = field(default="", init=False, repr=False)
    _spoke_pending: bool = field(default=False, init=False, repr=False)
    _base_k: int = field(default=0, init=False, repr=False)
    _throttle_streak: int = field(default=0, init=False, repr=False)
    _escape_hatch_pending: bool = field(default=False, init=False, repr=False)
    _partner_trust_pending: float = field(default=0.0, init=False, repr=False)
    _partner_uncertainty_pending: float = field(default=0.0, init=False, repr=False)
    _partner_name_pending: str = field(default="", init=False, repr=False)
    _pause_s_pending: float = field(default=0.0, init=False, repr=False)
    _claim_conflict_pending: float = field(default=0.0, init=False, repr=False)
    _metacog_conflict_pending: float = field(default=0.0, init=False, repr=False)
    _metacog_metastability_pending: float = field(default=0.0, init=False, repr=False)
    _metacog_saturation_pending: float = field(default=0.0, init=False, repr=False)
    _reset_level_pending: str = field(default="", init=False, repr=False)
    _change_kind_pending: str = field(default="", init=False, repr=False)
    _consolidated_pruned_pending: int = field(default=0, init=False, repr=False)
    _last_consolidation_tick: int = field(default=0, init=False, repr=False)
    _current_tick: int = field(default=0, init=False, repr=False)
    _probe_affordance_pending: str = field(default="", init=False, repr=False)
    _probe_success_pending: bool = field(default=False, init=False, repr=False)
    _actuation_status_pending: str = field(default="", init=False, repr=False)
    _actuation_goal_pending: str = field(default="", init=False, repr=False)
    _actuation_impatience_pending: float = field(default=0.0, init=False, repr=False)
    _actuation_steps_pending: int = field(default=0, init=False, repr=False)
    _actuation_preemptions_pending: int = field(default=0, init=False, repr=False)

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
        if self.escape_hatch_ticks < 0:
            raise ValueError(
                f"escape_hatch_ticks must be >= 0, got {self.escape_hatch_ticks}"
            )
        # Базовое k запоминается для восстановления после throttle
        # (throttle — обратимая регуляция, а не дрейф конфигурации).
        self._base_k = self.pipeline.voting.k
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

    def _gamma_bus(self, u: np.ndarray) -> np.ndarray:
        """Точность γ по каналам шины (без tile по колонкам).

        ``variance`` — γ = 1/var по окну (PrecisionEstimator).
        ``ones`` — baseline (фоллбэк FEP, ADR-0005 §4).

        Args:
            u: Вектор шины (с приором памяти) shape=(total_dim,).

        Returns:
            Вектор γ shape=(total_dim,).
        """
        if self.precision_mode == "ones":
            return np.ones(u.shape[0], dtype=np.float64)
        return self.estimator.update(u)

    def precision(self, u: np.ndarray) -> np.ndarray:
        """Точность γ для всех каналов.

        Args:
            u: Вектор шины (с приором памяти) shape=(total_dim,).

        Returns:
            Вектор γ shape=(N_columns * total_dim,).
        """
        return np.tile(self._gamma_bus(u), self.pipeline.ensemble.n_columns)

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
        # Ablation содержания памяти: recall выключен → канал остаётся, но
        # «молчит» (нулевой prior). Провайдер/размерность не меняются.
        self._cached_prior = (
            self.memory.recall_prior(query)
            if self.recall_enabled
            else np.zeros(self.memory.prior_dim, dtype=np.float64)
        )
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
        self._current_tick = tick
        dt, now = self._time_for_tick(tick)

        # Сенсорная фаза (включая эмбеддинг — сетевой I/O) НЕ входит в
        # ресурсную латентность: сеть — внешняя нагрузка, а «тахикардия»
        # измеряет собственные вычисления хоста.
        u_base = self.bus.step(tick, now)

        # Рефлекс (S4): критический интеро-сигнал → throttle в этом же тике,
        # минуя policy, dwell и attractor (манифест §3.К). Гомеостаз оценивает
        # сигналы прошлого тика; реакция ≤ 1 тик (VALIDATION §2.3).
        homeostasis = (
            self.homeostat.evaluate(self.bus.last_signals)
            if self.homeostat is not None
            else None
        )
        throttle = (
            plan_throttle(
                homeostasis,
                severity_threshold=self.throttle_severity_threshold,
                k_scale=self.throttle_k_scale,
                dt_scale=self.throttle_dt_scale,
            )
            if homeostasis is not None
            else ThrottlePlan(active=False)
        )
        if throttle.active:
            dt *= throttle.dt_scale
            k_eff = max(1, round(self._base_k * throttle.k_scale))
            self.pipeline.voting.set_k(k_eff)
            self._throttle_streak += 1
        else:
            # Обратимость: восстановить базовое k, когда сигнал нормализовался.
            self.pipeline.voting.set_k(self._base_k)
            self._throttle_streak = 0
        # Escape hatch (S4-долг): при удержании throttle дольше порога хост
        # получает право сообщить оператору о перегрузке, не жгя дорогой
        # инициативный LLM-вызов. Триггер — удержание, а не мгновение
        # (анти-дребезг). Отдельно от ``llm_gate``: инициатива запрещена,
        # escape hatch разрешён.
        self._escape_hatch_pending = (
            self.escape_hatch_ticks > 0
            and self._throttle_streak >= self.escape_hatch_ticks
        )

        text = self.message_provider.text_at(tick) if self.message_provider else ""
        has_new_message = bool(text) and text != self._last_text
        query = self._update_prior_if_new_text(text)
        prior = self._cached_prior
        u = np.concatenate([u_base, prior]) if prior is not None else u_base

        compute_start = time.perf_counter()
        gamma_bus = self._gamma_bus(u)
        u_eff = u
        if self.attention_gate:
            # Пред-колоночный барьер внимания (S4, ADR-0005 §2): доверие
            # каналу γ управляет прохождением входа. Обратимо: gate=False →
            # u_eff == u (контур S1–S3).
            weights = attention_gate(
                gamma_bus,
                gamma_ref=self.attention_gamma_ref,
                floor=self.attention_floor,
            )
            u_eff = apply_attention(u, weights)
        gamma = np.tile(gamma_bus, self.pipeline.ensemble.n_columns)
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
        outcome = self.pipeline.tick(
            u_eff, gamma, dt, segments, reflex_tags, self.importance
        )
        check_finite(outcome.result)
        drift = self.drift.update(outcome.result)
        self.meter.record_tick(time.perf_counter() - compute_start)

        # Selfcontrol (S6, read-only): наблюдаемые + план сброса. Не влияет на
        # F(t) того же тика (анти-circularity, ADR-0009 §1). Сбой → безопасный
        # дефолт, loop жив (инвариант 1).
        if self.selfcontrol is not None:
            metacognition, reset_plan = self.selfcontrol.observe(
                scores=outcome.activities
                if outcome.activities is not None
                else np.zeros(1),
                switched=outcome.switched,
                f=outcome.result.f,
                partner_uncertainty=self._partner_uncertainty_pending,
            )
            self._metacog_conflict_pending = metacognition.conflict
            self._metacog_metastability_pending = metacognition.metastability
            self._metacog_saturation_pending = metacognition.saturation
            self._reset_level_pending = (
                reset_plan.level.value if reset_plan.triggered else ""
            )
            assessment = self.selfcontrol.last_assessment
            self._change_kind_pending = (
                assessment.kind.value if assessment is not None else ""
            )

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

        # Ночной цикл (S6 проход 2): консолидация по расписанию/объёму.
        # Вне аффективного контура: не влияет на F этого тика (инвариант 6).
        self._maybe_consolidate(tick, now)

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
            throttle=throttle.active,
            homeostasis=(homeostasis.max_deviation if homeostasis is not None else 0.0),
            policy_action=self._policy_action_pending,
            policy_reason=self._policy_reason_pending,
            escape_hatch=self._escape_hatch_pending,
            partner_trust=self._partner_trust_pending,
            partner_uncertainty=self._partner_uncertainty_pending,
            partner_name=self._partner_name_pending,
            pause_s=self._pause_s_pending,
            claim_conflict=self._claim_conflict_pending,
            metacog_conflict=self._metacog_conflict_pending,
            metacog_metastability=self._metacog_metastability_pending,
            metacog_saturation=self._metacog_saturation_pending,
            reset_level=self._reset_level_pending,
            change_kind=self._change_kind_pending,
            consolidated_pruned=self._consolidated_pruned_pending,
            probe_affordance=self._probe_affordance_pending,
            probe_success=self._probe_success_pending,
            actuation_status=self._actuation_status_pending,
            actuation_goal=self._actuation_goal_pending,
            actuation_impatience=self._actuation_impatience_pending,
            actuation_steps=self._actuation_steps_pending,
            actuation_preemptions=self._actuation_preemptions_pending,
            channel_contrib=",".join(
                f"{tag}:{value:.4f}" for tag, value in outcome.channel_contrib
            ),
        )
        self._spoke_pending = False
        self._policy_action_pending = ""
        self._policy_reason_pending = ""
        self._consolidated_pruned_pending = 0
        self._probe_affordance_pending = ""
        self._probe_success_pending = False
        self._actuation_status_pending = ""
        self._actuation_goal_pending = ""
        self._actuation_impatience_pending = 0.0
        self._actuation_steps_pending = 0
        self._actuation_preemptions_pending = 0
        self.last_outcome = outcome
        self.last_drift = drift
        self.last_memory_hit = memory_hit
        self.last_homeostasis = homeostasis
        self.last_throttle = throttle
        return outcome

    def mark_spoke(self) -> None:
        """Отметить, что хост сгенерировал реплику (попадёт в телеметрию)."""
        self._spoke_pending = True

    @property
    def current_tick(self) -> int:
        """Номер последнего выполненного тика (для диагностического снимка)."""
        return self._current_tick

    @property
    def last_reset_level(self) -> str:
        """Уровень сброса последнего тика ("", soft/freeze/hard) — S6."""
        return self._reset_level_pending

    @property
    def last_change_kind(self) -> str:
        """Классификация изменения последнего тика (S6)."""
        return self._change_kind_pending

    @property
    def last_partner_trust(self) -> float:
        """Доверие к партнёру, зафиксированное для следующего тика (S5)."""
        return self._partner_trust_pending

    @property
    def last_partner_uncertainty(self) -> float:
        """Неопределённость идентичности партнёра (S5)."""
        return self._partner_uncertainty_pending

    @property
    def last_metacog_conflict(self) -> float:
        """Несогласие ансамбля колонок последнего тика (S6)."""
        return self._metacog_conflict_pending

    @property
    def last_metacog_metastability(self) -> float:
        """Метастабильность аттрактора последнего тика (S6)."""
        return self._metacog_metastability_pending

    @property
    def last_metacog_saturation(self) -> float:
        """Насыщение/тренд F последнего тика (S6)."""
        return self._metacog_saturation_pending

    @property
    def escape_hatch_active(self) -> bool:
        """Разрешён ли escape hatch: throttle удерживается ≥ порога тиков.

        Право сообщить оператору о перегрузке без дорогого LLM-вызова
        (S4-долг). Не путать с ``last_throttle.llm_gate``: инициатива
        запрещена, escape hatch — разрешён.
        """
        return self._escape_hatch_pending

    def policy_context(
        self,
        *,
        has_new_message: bool,
        mode: str = "free",
        partner: PartnerView | None = None,
    ) -> PolicyContext:
        """Собрать расширяемый контекст policy из состояния хоста.

        Policy-слой (S4) решает речевое действие вне тика; loop поставляет
        ему актуальное состояние. Метакогниция (S6) добавится новым полем
        ``PolicyContext`` без изменения этого метода. ``partner`` (S5) —
        модель партнёра из ``tm``; ``None`` → S4-совместимость.

        Args:
            has_new_message: Пришло ли новое сообщение оператора.
            mode: Режим хоста (макро-контекст).
            partner: Состояние партнёра (ToM, S5) или None.

        Returns:
            PolicyContext с F/аффектом/задачей/гомеостазом/партнёром.
        """
        outcome = self.last_outcome
        homeostasis = self.last_homeostasis
        if homeostasis is None:
            homeostasis = HomeostasisState(
                signals=(), max_deviation=0.0, severity=0.0, is_critical=False
            )
        macro = MacroContext(task=self.active_task(), mode=mode)
        return PolicyContext(
            f=outcome.result.f if outcome is not None else 0.0,
            valence=outcome.result.valence if outcome is not None else 0.0,
            stress=outcome.result.allostatic_stress if outcome is not None else 0.0,
            task=macro.task,
            homeostasis=homeostasis,
            has_new_message=has_new_message,
            mode=macro.mode,
            partner=partner,
            metacognition=(
                self.selfcontrol.metacognition if self.selfcontrol is not None else None
            ),
        )

    def active_task(self) -> str:
        """Тег активной задачи (специализация колонки-аттрактора)."""
        mask = self.pipeline.attractor.current_mask
        if mask is None:
            return "none"
        idx = int(np.argmax(mask))
        columns = self.pipeline.ensemble.column_configs
        if 0 <= idx < len(columns):
            return columns[idx].specialization
        return "none"

    def record_social(
        self,
        *,
        trust: float = 0.0,
        uncertainty: float = 0.0,
        name: str = "",
        pause_s: float = 0.0,
        claim_conflict: float = 0.0,
    ) -> None:
        """Зафиксировать социальные метрики для телеметрии следующего тика (S5).

        Args:
            trust: Доверие к партнёру, [0, 1].
            uncertainty: Неопределённость идентичности, [0, 1].
            name: Принятое имя партнёра ("" если нет).
            pause_s: Интервал с прошлой реплики, с.
            claim_conflict: Рассогласование последнего утверждения, [0, 1].
        """
        self._partner_trust_pending = trust
        self._partner_uncertainty_pending = uncertainty
        self._partner_name_pending = name
        self._pause_s_pending = pause_s
        self._claim_conflict_pending = claim_conflict

    def record_consolidation(self, pruned: int) -> None:
        """Зафиксировать число удалённых при консолидации эпизодов (S6).

        Args:
            pruned: Сколько эпизодов удалено (попадёт в телеметрию).
        """
        self._consolidated_pruned_pending = pruned

    def consolidate_memory(self) -> int:
        """Выполнить явную консолидацию памяти (S6) и вернуть число удалённых.

        Требует ``autonomy``-параметров и включённой памяти. Удаление логируется
        (инвариант 6). Возвращает 0, если память/автономия выключены.

        Returns:
            Число удалённых эпизодов.
        """
        return self._consolidate(self.clock(), self._current_tick)

    def _consolidate(self, now: float, tick: int) -> int:
        """Исполнить консолидацию в момент ``now`` (Shell, общий путь).

        Args:
            now: Время для расчёта давности эпизодов.
            tick: Тик, к которому привязывается консолидация.

        Returns:
            Число удалённых эпизодов.
        """
        if self.memory is None or self.autonomy_config is None:
            return 0
        from src.memory import consolidate

        result = consolidate(
            self.memory.store,
            min_weight=self.autonomy_config.consolidate_min_weight,
            schema_threshold=self.autonomy_config.schema_threshold,
            max_schemas=self.autonomy_config.max_schemas,
            now=now,
            recency_tau_s=self.autonomy_config.recency_tau_s,
        )
        self.record_consolidation(result.pruned)
        self._last_consolidation_tick = tick
        return result.pruned

    def _maybe_consolidate(self, tick: int, now: float) -> None:
        """Запустить ночной цикл, если пора (S6 проход 2).

        Триггер — по расписанию (``consolidate_every_ticks``) и объёму
        (``consolidate_min_episodes``). Вызывается в конце тика: удаление
        логируется в телеметрию **этого** тика (инвариант 6). Выключено при
        ``every_ticks == 0`` (S5/S6-проход-1-совместимость). Время берётся из
        источника loop (``now``), чтобы synthetic-прогон оставался
        детерминированным.

        Args:
            tick: Текущий тик.
            now: Время текущего тика (из ``_time_for_tick``).
        """
        if self.memory is None or self.autonomy_config is None:
            return
        trigger = should_consolidate(
            tick=tick,
            last_tick=self._last_consolidation_tick,
            episode_count=self.memory.episode_count(),
            every_ticks=self.autonomy_config.consolidate_every_ticks,
            min_episodes=self.autonomy_config.consolidate_min_episodes,
        )
        if trigger.due:
            self._consolidate(now, tick)

    def explore(self, *, reason: str = "epistemic drive") -> ProbeResult | None:
        """Выполнить эпистемическое зондирование (S6 проход 2).

        Мягкий драйв: зондируем только если метакогнитивная неопределённость
        выше порога пресета и доступен обратимый аффорданс. Исполнение идёт
        через capability gate (fail-safe deny). Результат попадёт в телеметрию
        следующего тика (как ``record_policy``).

        Args:
            reason: Причина зондирования (для аудита).

        Returns:
            ProbeResult или None (нет effector'а / неопределённость ниже порога).
        """
        if self.probe_effector is None:
            return None
        uncertainty = (
            self.selfcontrol.metacognition.epistemic_uncertainty
            if self.selfcontrol is not None
            and self.selfcontrol.metacognition is not None
            else 0.0
        )
        threshold = (
            self.autonomy_config.explore_threshold
            if self.autonomy_config is not None
            else 1.0
        )
        affordance = select_affordance(
            uncertainty, self.probe_effector.affordances, threshold=threshold
        )
        if affordance is None:
            return None
        result = self.probe_effector.probe(
            ProbeRequest(affordance=affordance.name, reason=reason)
        )
        self._probe_affordance_pending = result.affordance
        self._probe_success_pending = result.success
        return result

    def record_policy(self, trace: PolicyTrace) -> None:
        """Зафиксировать решение policy для телеметрии следующего тика.

        Args:
            trace: Причинная трасса решения (explainability, S4).
        """
        self.last_policy_trace = trace
        self._policy_action_pending = trace.chosen.value
        self._policy_reason_pending = trace.reason

    def tick_actuation(
        self,
        root: Node,
        facts: Mapping[str, float] | None = None,
    ) -> ExecutorOutcome | None:
        """Провести тик секвенирования актуаций (S8 этап 7).

        Вызывается Shell'ом (chat/behavioral) поверх ``step_once``: ведёт
        выведенное дерево ``root`` во времени через эффекторы. Секвенирование
        выключено (``executor is None``) → no-op, поля телеметрии пусты → контур
        S7 идентичен. Завершённые активации возвращаются в ``outcome.completed``
        (→ шина, ADR-0012 §7).

        Args:
            root: Корень текущего deliberative-дерева (из генератора, этап 5).
            facts: Снимок фактов мира (для кондишенов); None → пусто.

        Returns:
            ExecutorOutcome текущего тика или None (секвенирование выключено).
        """
        if self.executor is None:
            return None
        outcome = self.executor.tick(root, dict(facts) if facts else {})
        self._actuation_status_pending = outcome.status.value
        self._actuation_goal_pending = outcome.running_goal or ""
        self._actuation_impatience_pending = outcome.impatience
        self._actuation_steps_pending = len(outcome.completed)
        self._actuation_preemptions_pending = outcome.preemptions
        return outcome

    @property
    def actuation_enabled(self) -> bool:
        """Включено ли секвенирование актуаций (S8)."""
        return self.executor is not None

    @property
    def last_actuation_goal(self) -> str:
        """Бегущая активация последнего тика ("" — ничего не бежит) — S8."""
        return self._actuation_goal_pending

    @property
    def last_actuation_telemetry(self) -> tuple[str, str, float, int, int]:
        """Поля телеметрии актуаций последнего тика (S8).

        Returns:
            (status, goal, impatience, steps, preemptions).
        """
        return (
            self._actuation_status_pending,
            self._actuation_goal_pending,
            self._actuation_impatience_pending,
            self._actuation_steps_pending,
            self._actuation_preemptions_pending,
        )

    def attach_speech(self, speak: Callable[[Actuation], str | None]) -> None:
        """Подключить речевой эффектор к секвенированию (S8, Shell).

        Речь доступна только в диалоговом стенде (``SpeechController`` создаётся
        после loop), поэтому эффектор подключается отдельно. Секвенирование
        выключено (``executor is None``) → no-op.

        Args:
            speak: Функция генерации речи (актуация → текст или None).
        """
        if self.executor is None:
            return
        latency = self.actuation_config.latency_ticks if self.actuation_config else 0
        self.executor.effectors[ActuationKind.SPEAK] = SpeechEffector(
            speak, latency_ticks=latency
        )

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
    integrations: IntegrationRegistry | None = None,
    probe_fn: ProbeFn | None = None,
    affordances: AffordanceMap | None = None,
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
        integrations: Реестр интеграций (ADR-0011). None → карта аффордансов
            из ``default_affordances()`` (совместимость S6).
        probe_fn: Реальный транспорт зондирования (ADR-0011). None → mock.
        affordances: Готовая карта аффордансов (приоритетнее реестра). None →
            из реестра, иначе ``default_affordances()``.

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
            api_dim=int(api["api_dim"]),  # type: ignore[call-overload]
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

    # Важность каналов (BACKLOG): rank₀ → per-component wᵢ = rank/dim.
    # Пусто → None (legacy: F = 0.5·Σγ·e², обратная совместимость S1–S8).
    # Веса включаются только при явном объявлении рангов оператором.
    importance: np.ndarray | None = None
    if config.channel_ranks:
        importance_segments = bus.segments
        if prior_dim > 0:
            importance_segments = importance_segments + (
                BusSegment(name="memory", offset=bus.bus_dim, dim=prior_dim, period=1),
            )
        importance = channel_importance(
            importance_segments,
            dict(config.channel_ranks),
            total_dim=total_dim,
        )

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

    homeostat = Homeostat(
        setpoints=config.homeostasis.setpoints,
        reflex_threshold=config.homeostasis.reflex_threshold,
    )

    selfcontrol: SelfMonitor | None = None
    probe_effector: ProbeEffector | None = None
    if config.autonomy.enabled:
        selfcontrol = SelfMonitor(
            window=config.autonomy.metacog_window,
            variance_gain=config.autonomy.csd_variance_gain,
            autocorr_gain=config.autonomy.csd_autocorr_gain,
            warning_threshold=config.autonomy.csd_warning_threshold,
            soft_threshold=config.autonomy.reset_soft_threshold,
        )
        # MCP-зондирование (S6 проход 2): карта аффордансов + gate. Права —
        # обратимое действие автономно (T3), необратимое — с HITL (T4).
        # ADR-0011: карта берётся из реестра интеграций, если он передан;
        # иначе — прежний default_affordances() (совместимость S6).
        if affordances is None:
            if integrations is not None:
                affordances = to_affordances(integrations.specs)
            else:
                affordances = default_affordances()
        probe_effector = ProbeEffector(
            affordances=affordances,
            gate=CapabilityGate(
                max_tier=CapabilityTier.T4,
                granted=frozenset(
                    {
                        Capability.READ,
                        Capability.ACT_REVERSIBLE,
                        Capability.ACT_IRREVERSIBLE,
                    }
                ),
            ),
            probe_fn=probe_fn,
        )

    # Секвенирование актуаций (S8 этап 7): executor над эффекторами. Дефолт
    # enabled=False → S7-совместимость (речь как была, без BT-контура). Тул-
    # эффектор подключается лишь при доступном probe_effector (иначе гейта нет
    # и вызовы молча «проходили» бы — так делать нельзя). Речевой эффектор
    # подключается Shell'ом позже (`attach_speech`), когда есть SpeechController.
    executor: ActuatorExecutor | None = None
    if config.actuation.enabled:
        effectors: dict[ActuationKind, Effector] = {}
        if probe_effector is not None:
            effectors[ActuationKind.INVOKE_TOOL] = ToolEffector(
                probe_effector, latency_ticks=config.actuation.latency_ticks
            )
        executor = ActuatorExecutor(
            effectors=effectors,
            expected_ticks=config.actuation.expected_ticks,
        )

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
        homeostat=homeostat,
        throttle_k_scale=config.homeostasis.throttle_k_scale,
        throttle_dt_scale=config.homeostasis.throttle_dt_scale,
        throttle_severity_threshold=config.homeostasis.reflex_threshold,
        escape_hatch_ticks=config.homeostasis.escape_hatch_ticks,
        attention_gate=config.policy.attention_gate,
        policy_config=config.policy,
        tick_dt=config.dt,
        clock_mode=config.clock_mode,
        paced=config.paced,
        precision_mode=config.precision_mode,
        time_scale=config.time_scale,
        memory=memory,
        message_provider=message_provider,
        recall_enabled=config.memory.recall_enabled,
        importance=importance,
        selfcontrol=selfcontrol,
        autonomy_config=config.autonomy,
        probe_effector=probe_effector,
        actuation_config=config.actuation,
        executor=executor,
    )
