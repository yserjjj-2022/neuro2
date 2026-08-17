"""Wiring — connects independently developed modules into a working pipeline.

Builds the energy observer wired to a file-based telemetry logger.
This is the ONLY place in the codebase that knows the concrete field
names of FreeEnergyResult (result.f, result.valence, etc.),
isolating the risk of interface desync to one location.

Also builds the full per-tick pipeline: CMC → voting → energy → telemetry.
CMCPipeline.tick(u, precision) runs one complete host tick; u and precision
are passed by the caller (future host loop / Phase 2 MCP resources).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.cmc import CMCEnsemble, ColumnConfig
from src.core.cmc.models import Vector
from src.core.energy import EnergyObserver, FreeEnergyCalculator, FreeEnergyResult
from src.core.voting import VotingManager
from src.telemetry import TelemetryLogger, TelemetryWriter


@dataclass(frozen=True)
class CMCPipeline:
    """Композиция per-tick: CMC → voting → energy → telemetry.

    Attributes:
        ensemble: Ансамбль колонок (производитель e(t) и активностей).
        voting: k-WTA по активностям колонок (результат кэшируется в .last
            для будущих аттракторов задач — Фаза 2).
        observer: EnergyObserver с sink в телеметрию (active_columns
            берётся из ensemble.active — закрывает TODO из build_energy_pipeline).
    """

    ensemble: CMCEnsemble
    voting: VotingManager
    observer: EnergyObserver

    def tick(self, u: Vector, precision: Vector) -> FreeEnergyResult:
        """Один полный тик хоста: CMC → voting → energy → telemetry.

        Args:
            u: Вход ансамбля L4, shape == (input_dim,).
            precision: Вектор точности γ для energy, shape == raveled errors
                (N_columns * input_dim).

        Returns:
            FreeEnergyResult — метрики текущего тика.

        Raises:
            ValueError: Если u.shape != (input_dim,) или
                precision.shape != (N_columns * input_dim,) — fail-fast,
                до вызова observer (понятная ошибка на уровне конвейера,
                а не внутри FreeEnergyCalculator).
        """
        out = self.ensemble.step(u)

        expected = np.ravel(out.errors).shape
        if precision.shape != expected:
            raise ValueError(
                f"Shape mismatch: precision {precision.shape} != "
                f"raveled errors {expected} (N_columns * input_dim)"
            )

        # Активности колонок = ‖e‖² по строкам errors → вход для k-WTA
        activities = np.sum(out.errors**2, axis=1)
        self.voting.vote(activities)  # результат → .last (телеметрия/логирование)

        return self.observer.observe(np.ravel(out.errors), precision)


def build_energy_pipeline(log_path: Path) -> EnergyObserver:
    """Собирает observer, подключённый к файловому логгеру.

    Args:
        log_path: Путь к JSONL-файлу для записи телеметрии.

    Returns:
        Настроенный EnergyObserver с sink, который пишет в указанный файл.
    """
    writer = TelemetryWriter(log_path=log_path)
    telemetry_logger = TelemetryLogger(writer=writer, phase="phase1", mode="free")

    def sink(result: FreeEnergyResult) -> None:
        telemetry_logger.log(
            free_energy=result.f,
            valence=result.valence,
            allostatic_stress=result.allostatic_stress,
            # active_columns=0 (default) — TODO(Phase 3): из src/core/cmc/
        )

    return EnergyObserver(calculator=FreeEnergyCalculator(), sink=sink)


def build_cmc_pipeline(
    columns: list[ColumnConfig],
    k: int,
    log_path: Path,
    active_threshold: float = 0.0,
) -> CMCPipeline:
    """Собирает полный per-tick конвейер: CMC → voting → energy → telemetry.

    Args:
        columns: Конфигурации колонок ансамбля (единые input_dim/state_dim).
        k: Число победителей k-WTA (1 = hard-WTA).
        log_path: Путь к JSONL-файлу для записи телеметрии.
        active_threshold: Порог активности колонки (‖e‖² > threshold).

    Returns:
        CMCPipeline — готовый к tick(u, precision).
    """
    ensemble = CMCEnsemble(columns=columns, active_threshold=active_threshold)
    voting = VotingManager(k=k)

    writer = TelemetryWriter(log_path=log_path)
    telemetry_logger = TelemetryLogger(writer=writer, phase="phase1", mode="free")

    def sink(result: FreeEnergyResult) -> None:
        # active_columns — результат последнего step() (ensemble.active
        # обновляется в tick до вызова observe(), т.е. sink видит текущий тик)
        telemetry_logger.log(
            free_energy=result.f,
            valence=result.valence,
            allostatic_stress=result.allostatic_stress,
            active_columns=ensemble.active,
        )

    observer = EnergyObserver(calculator=FreeEnergyCalculator(), sink=sink)
    return CMCPipeline(ensemble=ensemble, voting=voting, observer=observer)
