"""Host parameters — single source of truth for tunable values.

CONSTITUTION §2.2: all temporal magnitudes and thresholds live here, not
hardcoded in logic modules. Values are STARTING points (test-tuned), not
calibrated: threshold calibration by telemetry is Phase 2/3 (BACKLOG).

Naming mirrors the constructor arguments of the modules that consume them,
so wiring is a direct mapping without translation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.attractors import TaskAttractor
from src.core.cmc import ColumnConfig
from src.core.energy import FreeEnergyCalculator
from src.core.homeostasis import Setpoint
from src.core.policy import Preferences


@dataclass(frozen=True)
class MemoryConfig:
    """Параметры эпизодической памяти и эмбеддера (S2).

    Настройки API-эмбеддера (base_url, model, dim) — в окружении (.env):
    ``EMBEDDER_BASE_URL``, ``EMBEDDER_MODEL``, ``EMBEDDER_DIM``, ``EMBEDDER_API_KEY``
    (RouterAI по умолчанию). Здесь — поведение памяти и размерность fake.

    Attributes:
        enabled: Включать ли память в host loop.
        embedder_mode: "auto" (ключ→api, иначе fake), "fake", "api".
        embedding_dim: Размерность fake-эмбеддера (= dim коммуникативного входа).
        db_path: Путь к SQLite-файлу памяти.
        episode_spike_threshold: Порог всплеска F для записи эпизода.
        recall_limit: Сколько эпизодов извлекать при recall.
        prior_dim: Размерность приора памяти в шине.
    """

    enabled: bool = True
    embedder_mode: str = "auto"
    embedding_dim: int = 8
    db_path: str = "host_memory.db"
    episode_spike_threshold: float = 1.0
    recall_limit: int = 1
    prior_dim: int = 4

    def __post_init__(self) -> None:
        """Валидация: положительные размеры, известный режим, пороги."""
        if self.embedder_mode not in ("auto", "fake", "api"):
            raise ValueError(
                f"embedder_mode must be 'auto'|'fake'|'api', got {self.embedder_mode!r}"
            )
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {self.embedding_dim}")
        if self.prior_dim <= 0:
            raise ValueError(f"prior_dim must be > 0, got {self.prior_dim}")
        if self.recall_limit < 1:
            raise ValueError(f"recall_limit must be >= 1, got {self.recall_limit}")
        if self.episode_spike_threshold < 0.0:
            raise ValueError(
                f"episode_spike_threshold must be >= 0, "
                f"got {self.episode_spike_threshold}"
            )


@dataclass(frozen=True)
class EnergyConfig:
    """Параметры FreeEnergyCalculator.

    Attributes:
        stress_leak_per_sec: λ — утечка аллостатического стресса, 1/с.
        valence_tau: τ — постоянная времени сглаживания валентности, с.
        gamma_base: γ по умолчанию для пустого входа.
    """

    stress_leak_per_sec: float = 0.01
    valence_tau: float = 1.0
    gamma_base: float = 1.0

    def build(self) -> FreeEnergyCalculator:
        """Собрать калькулятор с этими параметрами.

        Returns:
            FreeEnergyCalculator.
        """
        return FreeEnergyCalculator(
            stress_leak_per_sec=self.stress_leak_per_sec,
            valence_tau=self.valence_tau,
            gamma_base=self.gamma_base,
        )


@dataclass(frozen=True)
class ColumnParams:
    """Параметры одной колонки.

    Attributes:
        specialization: Тег специализации ("tone", "rhythm", ...).
        alpha: Скорость обновления состояния, α ∈ [0, 1].
    """

    specialization: str = "general"
    alpha: float = 0.1

    def build(self, input_dim: int, state_dim: int) -> ColumnConfig:
        """Собрать ColumnConfig под конкретные размерности шины.

        Args:
            input_dim: Размерность входа L4 (= ширина шины в Фазе 1).
            state_dim: Размерность состояния L5/6 (== input_dim в Фазе 1).

        Returns:
            ColumnConfig.
        """
        return ColumnConfig(
            input_dim=input_dim,
            state_dim=state_dim,
            specialization=self.specialization,
            alpha=self.alpha,
        )


@dataclass(frozen=True)
class AttractorConfig:
    """Параметры TaskAttractor (dwell, basin, convergence, dominance).

    Attributes:
        base_dwell: α — жёсткий пол удержания, тики.
        dwell_slope: β — прирост dwell при выигрыше.
        plasticity_gain: γ — прирост устойчивости за тик удержания (STP).
        basin_threshold: ε — порог бассейна притяжения.
        convergence_threshold: τ — порог сходимости EMA.
        dominance_threshold: δ — порог явного превосходства конкурента.
    """

    base_dwell: int = 5
    dwell_slope: float = 2.0
    plasticity_gain: float = 0.1
    basin_threshold: float = 0.15
    convergence_threshold: float = 1e-8
    dominance_threshold: float = 0.3

    def build(self, n_tasks: int) -> TaskAttractor:
        """Собрать аттрактор на N задач.

        Args:
            n_tasks: Число колонок (задач).

        Returns:
            TaskAttractor.
        """
        return TaskAttractor(
            n_tasks=n_tasks,
            base_dwell=self.base_dwell,
            dwell_slope=self.dwell_slope,
            plasticity_gain=self.plasticity_gain,
            basin_threshold=self.basin_threshold,
            convergence_threshold=self.convergence_threshold,
            dominance_threshold=self.dominance_threshold,
        )


@dataclass(frozen=True)
class SpeechConfig:
    """Параметры речи (S3).

    Настройки LLM (base_url, model) — в окружении (.env): ``LLM_BASE_URL``,
    ``LLM_MODEL``, ``LLM_API_KEY`` (RouterAI по умолчанию).

    Attributes:
        enabled: Включать ли речь (S3 по умолчанию выкл; включается --chat).
        llm_mode: "auto" (ключ→api, иначе fake), "fake", "api".
        f_threshold: Порог F для инициативы (не для ответа на сообщение).
        recall_limit: Сколько прецедентов подавать в Intent-Frame.
        default_register: Речевой режим (brief/terse/normal/story).
        history_turns: Глубина истории диалога (сообщений).
        temperature: Температура генерации.
        style: Дефолтный стиль (S5 — из характера).
        reasoning: Включить reasoning у LLM (по умолчанию False; ADR-0007).
    """

    enabled: bool = False
    llm_mode: str = "auto"
    f_threshold: float = 1.0
    recall_limit: int = 3
    default_register: str = "brief"
    history_turns: int = 20
    temperature: float = 0.7
    style: str = "neutral"
    reasoning: bool = False

    def __post_init__(self) -> None:
        """Валидация: известные режимы, неотрицательные пороги."""
        if self.llm_mode not in ("auto", "fake", "api"):
            raise ValueError(
                f"llm_mode must be 'auto'|'fake'|'api', got {self.llm_mode!r}"
            )
        if self.f_threshold < 0.0:
            raise ValueError(f"f_threshold must be >= 0, got {self.f_threshold}")
        if self.recall_limit < 1:
            raise ValueError(f"recall_limit must be >= 1, got {self.recall_limit}")
        if self.default_register not in ("brief", "terse", "normal", "story"):
            raise ValueError(
                f"default_register must be brief|terse|normal|story, "
                f"got {self.default_register!r}"
            )
        if self.history_turns < 0:
            raise ValueError(f"history_turns must be >= 0, got {self.history_turns}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")


@dataclass(frozen=True)
class HomeostasisConfig:
    """Параметры гомеостаза (S4).

    Attributes:
        setpoints: Сетпоинты интероцептивных каналов (battery, resources, cpu).
        reflex_threshold: Порог severity для критического сигнала/рефлекса.
        throttle_k_scale: Множитель k-WTA при throttle, (0, 1].
        throttle_dt_scale: Множитель dt при throttle, >= 1.
        escape_hatch_ticks: Сколько тиков удержания throttle даёт право
            сообщить оператору о перегрузке (0 → escape hatch выключен).
    """

    setpoints: tuple[Setpoint, ...] = (
        Setpoint(tag="battery", comfort=0.5, critical=0.9),
        Setpoint(tag="resources", comfort=0.5, critical=0.9),
        Setpoint(tag="cpu", comfort=0.7, critical=0.9),
    )
    reflex_threshold: float = 0.9
    throttle_k_scale: float = 0.5
    throttle_dt_scale: float = 2.0
    escape_hatch_ticks: int = 3

    def __post_init__(self) -> None:
        """Валидация: непустые сетпоинты, границы порогов.

        Raises:
            ValueError: Если setpoints пуст, reflex_threshold вне [0, 1],
                throttle_k_scale вне (0, 1] или throttle_dt_scale < 1.
        """
        if not self.setpoints:
            raise ValueError("setpoints must not be empty")
        if not 0.0 <= self.reflex_threshold <= 1.0:
            raise ValueError(
                f"reflex_threshold must be in [0, 1], got {self.reflex_threshold}"
            )
        if not 0.0 < self.throttle_k_scale <= 1.0:
            raise ValueError(
                f"throttle_k_scale must be in (0, 1], got {self.throttle_k_scale}"
            )
        if self.throttle_dt_scale < 1.0:
            raise ValueError(
                f"throttle_dt_scale must be >= 1, got {self.throttle_dt_scale}"
            )
        if self.escape_hatch_ticks < 0:
            raise ValueError(
                f"escape_hatch_ticks must be >= 0, got {self.escape_hatch_ticks}"
            )


@dataclass(frozen=True)
class PolicyConfig:
    """Параметры policy (S4).

    Attributes:
        enabled: Включать ли policy (иначе S3-поведение: should_speak).
        preferences: Предпочитаемые исходы (goal-directed).
        mode: Режим хоста (макро-контекст): game/cooperative/free.
        attention_gate: Пред-колоночная γ (проход 2; S4 — выкл).
    """

    enabled: bool = True
    preferences: Preferences = field(default_factory=Preferences)
    mode: str = "free"
    attention_gate: bool = False

    def __post_init__(self) -> None:
        """Валидация режима хоста.

        Raises:
            ValueError: Если mode не из {game, cooperative, free}.
        """
        if self.mode not in ("game", "cooperative", "free"):
            raise ValueError(f"mode must be game|cooperative|free, got {self.mode!r}")


@dataclass(frozen=True)
class SocialConfig:
    """Параметры социального контура (S5, ToM).

    Attributes:
        enabled: Включать ли модель партнёра (иначе S4-совместимость).
        match_threshold: Порог косинусной близости для узнавания.
        signature_learning_rate: Скорость обновления сигнатуры, (0, 1].
        trust_gain: Прирост доверия при согласии, >= 0.
        trust_decay: Утечка доверия, >= 0.
        conflict_threshold: Порог рассогласования для гипотезы (Vigilance).
        pause_tau_s: Постоянная нормировки паузы диалога, с (> 0).
        identify_threshold: Порог uncertainty для мягкого интента.
    """

    enabled: bool = False
    match_threshold: float = 0.75
    signature_learning_rate: float = 0.2
    trust_gain: float = 0.1
    trust_decay: float = 0.01
    conflict_threshold: float = 0.6
    pause_tau_s: float = 5.0
    identify_threshold: float = 0.7

    def __post_init__(self) -> None:
        """Валидация: пороги в [0, 1], положительные tau/learning_rate.

        Raises:
            ValueError: При выходе параметров за допустимые границы.
        """
        for name in ("match_threshold", "conflict_threshold", "identify_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not 0.0 < self.signature_learning_rate <= 1.0:
            raise ValueError(
                f"signature_learning_rate must be in (0, 1], "
                f"got {self.signature_learning_rate}"
            )
        if self.trust_gain < 0.0 or self.trust_decay < 0.0:
            raise ValueError(
                f"trust_gain/trust_decay must be >= 0, "
                f"got {self.trust_gain}/{self.trust_decay}"
            )
        if self.pause_tau_s <= 0.0:
            raise ValueError(f"pause_tau_s must be > 0, got {self.pause_tau_s}")


@dataclass(frozen=True)
class HostConfig:
    """Полная конфигурация host loop.

    Attributes:
        dt: Шаг интегрирования loop в секундах (> 0). Дефолт 0.1 (10 Гц,
            эмоциональный контур, ADR-0006).
        max_ticks: Число тиков (0 → бесконечно, до Ctrl+C).
        k: Число победителей k-WTA.
        seed: Зерно детерминированных провайдеров.
        active_threshold: Порог активности колонки (‖e‖² > threshold).
        precision_mode: "variance" (γ=1/var) или "ones" (baseline).
        precision_window: Окно оценки дисперсии, тики.
        gamma_max: Потолок γ (во сколько раз максимум доверяем каналу).
        precision_eps: Регуляризация знаменателя γ.
        clock_mode: "synthetic" (tick·dt) или "wall" (реальное время).
        paced: Спать между тиками (реальное время ≈ dt).
        time_scale: Множитель субъективного времени (1.0 = жизнь, >1 = симуляция).
        tick_budget_ms: Бюджет длительности тика, мс.
        rss_budget_mb: Бюджет памяти, МБ.
        drift_f_threshold: Порог F для детектора дрейфа.
        drift_stress_threshold: Порог стресса для детектора дрейфа.
        log_path: Путь к JSONL-файлу телеметрии.
        columns: Параметры колонок.
        energy: Параметры energy.
        attractor: Параметры аттрактора.
        memory: Параметры памяти.
        speech: Параметры речи (S3).
        homeostasis: Параметры гомеостаза (S4).
        policy: Параметры policy (S4).
    """

    dt: float = 0.1
    max_ticks: int = 100
    k: int = 2
    seed: int = 0
    active_threshold: float = 1e-8
    precision_mode: str = "variance"
    precision_window: int = 50
    gamma_max: float = 10.0
    precision_eps: float = 1e-6
    clock_mode: str = "synthetic"
    paced: bool = False
    time_scale: float = 1.0
    tick_budget_ms: float = 50.0
    rss_budget_mb: float = 1024.0
    drift_f_threshold: float = 100.0
    drift_stress_threshold: float = 50.0
    log_path: str = "host_telemetry.jsonl"
    columns: tuple[ColumnParams, ...] = field(
        default_factory=lambda: (
            ColumnParams(specialization="tone"),
            ColumnParams(specialization="rhythm"),
            ColumnParams(specialization="meaning"),
        )
    )
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    attractor: AttractorConfig = field(default_factory=AttractorConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    homeostasis: HomeostasisConfig = field(default_factory=HomeostasisConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    social: SocialConfig = field(default_factory=SocialConfig)

    def __post_init__(self) -> None:
        """Валидация: положительные размеры, известные режимы."""
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.precision_mode not in ("ones", "variance"):
            raise ValueError(
                f"precision_mode must be 'ones' or 'variance', "
                f"got {self.precision_mode!r}"
            )
        if self.precision_window < 1:
            raise ValueError(
                f"precision_window must be >= 1, got {self.precision_window}"
            )
        if self.gamma_max <= 0.0:
            raise ValueError(f"gamma_max must be > 0, got {self.gamma_max}")
        if self.precision_eps <= 0.0:
            raise ValueError(f"precision_eps must be > 0, got {self.precision_eps}")
        if self.clock_mode not in ("synthetic", "wall"):
            raise ValueError(
                f"clock_mode must be 'synthetic' or 'wall', got {self.clock_mode!r}"
            )
        if self.time_scale <= 0.0:
            raise ValueError(f"time_scale must be > 0, got {self.time_scale}")
        if self.tick_budget_ms <= 0.0:
            raise ValueError(f"tick_budget_ms must be > 0, got {self.tick_budget_ms}")
        if self.rss_budget_mb <= 0.0:
            raise ValueError(f"rss_budget_mb must be > 0, got {self.rss_budget_mb}")
        if len(self.columns) < self.k:
            raise ValueError(
                f"k={self.k} exceeds number of columns {len(self.columns)}"
            )
