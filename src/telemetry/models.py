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
        tick: Номер тика хоста (S1: сопоставление строки с тиком).
        free_energy: Сырое значение F(t).
        valence: Валентность (-dF/dt, сглаженная).
        allostatic_stress: Интеграл F(t) по времени.
        gamma: Precision weighting γ (агрегат).
        active_columns: Количество активных колонок.
        active_tags: CSV-теги активных каналов шины.
        reflex_tags: CSV-теги критических (is_reflex) сигналов.
        bus_dim: Ширина шины (детект дрейфа конфигурации).
        latency_ms: Длительность тика, мс.
        rss_mb: RSS процесса, МБ.
        drift: Флаг детектора дрейфа (заготовка S1).
        phase: Фаза проекта (из config).
        mode: Режим (game/cooperative/free).
    """

    timestamp: float
    tick: int
    free_energy: float
    valence: float
    allostatic_stress: float
    gamma: float
    active_columns: int
    active_tags: str
    reflex_tags: str
    bus_dim: int
    latency_ms: float
    rss_mb: float
    drift: bool
    phase: str
    mode: str
