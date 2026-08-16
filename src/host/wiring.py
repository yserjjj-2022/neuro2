"""Wiring — connects independently developed modules into a working pipeline.

Builds the energy observer wired to a file-based telemetry logger.
This is the ONLY place in the codebase that knows the concrete field
names of FreeEnergyResult (result.f, result.valence, etc.),
isolating the risk of interface desync to one location.

TODO: When src/core/cmc/ appears (Phase 3), replace active_columns=0
with the real value from the CMC layer.
"""

from __future__ import annotations

from pathlib import Path

from src.core.energy import EnergyObserver, FreeEnergyCalculator, FreeEnergyResult
from src.telemetry import TelemetryLogger, TelemetryWriter


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
