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
class HostConfig:
    """Полная конфигурация host loop.

    Attributes:
        dt: Шаг интегрирования loop в секундах (> 0). Дефолт 0.1 (10 Гц,
            эмоциональный контур, ADR-0006).
        max_ticks: Число тиков (0 → бесконечно, до Ctrl+C).
        k: Число победителей k-WTA.
        seed: Зерно детерминированных провайдеров.
        message_dim: Размерность заглушки коммуникативного сигнала.
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
    """

    dt: float = 0.1
    max_ticks: int = 100
    k: int = 2
    seed: int = 0
    message_dim: int = 8
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

    def __post_init__(self) -> None:
        """Валидация: положительные размеры, известные режимы."""
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.message_dim <= 0:
            raise ValueError(f"message_dim must be > 0, got {self.message_dim}")
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
