"""Models for telemetry module.

Flat, serializable data structures for host state logging.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TelemetryEvent:
    """Событие телеметрии — плоская структура, сериализуемая в JSON.

    Аналог FreeEnergyResult из src/core/energy/, но для логирования.
    phase/mode заполняются TelemetryLogger.log(), а не вызывающим кодом.

    Attributes:
        timestamp: Unix timestamp (time.time()).
        free_energy: Сырое значение F(t).
        valence: Валентность (-dF/dt).
        allostatic_stress: Интеграл F(t) по времени.
        active_columns: Количество активных колонок.
        phase: Фаза проекта (из config).
        mode: Режим (game/cooperative/free).
    """

    timestamp: float
    free_energy: float
    valence: float
    allostatic_stress: float
    active_columns: int
    phase: str
    mode: str
