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
        dt: Шаг интегрирования для валентности (-dF/dt), секунды.
        stress_decay: Коэффициент затухания аллостатического стресса.
        gamma_base: γ по умолчанию для пустого входа.
    """

    dt: float = 0.01
    stress_decay: float = 0.99
    gamma_base: float = 1.0

    def build(self) -> FreeEnergyCalculator:
        """Собрать калькулятор с этими параметрами.

        Returns:
            FreeEnergyCalculator.
        """
        return FreeEnergyCalculator(
            dt=self.dt,
            stress_decay=self.stress_decay,
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
        dt: Шаг интегрирования loop в секундах (0 → без пауз).
        max_ticks: Число тиков (0 → бесконечно, до Ctrl+C).
        k: Число победителей k-WTA.
        seed: Зерно детерминированных провайдеров.
        message_dim: Размерность заглушки коммуникативного сигнала.
        active_threshold: Порог активности колонки (‖e‖² > threshold).
        precision_mode: "ones" (Фаза 1 baseline) или "variance" (Фаза 2).
        log_path: Путь к JSONL-файлу телеметрии.
        columns: Параметры колонок.
        energy: Параметры energy.
        attractor: Параметры аттрактора.
    """

    dt: float = 0.01
    max_ticks: int = 100
    k: int = 2
    seed: int = 0
    message_dim: int = 8
    active_threshold: float = 1e-8
    precision_mode: str = "ones"
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
        """Валидация: положительные размеры, известный режим precision."""
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.message_dim <= 0:
            raise ValueError(f"message_dim must be > 0, got {self.message_dim}")
        if self.dt < 0.0:
            raise ValueError(f"dt must be >= 0, got {self.dt}")
        if self.precision_mode not in ("ones", "variance"):
            raise ValueError(
                f"precision_mode must be 'ones' or 'variance', "
                f"got {self.precision_mode!r}"
            )
        if len(self.columns) < self.k:
            raise ValueError(
                f"k={self.k} exceeds number of columns {len(self.columns)}"
            )
