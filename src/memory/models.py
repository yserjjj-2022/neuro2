"""Episode — core domain object of the memory module.

Frozen dataclass, analogous to FreeEnergyResult (energy) and
TelemetryEvent (telemetry).
"""

from __future__ import annotations

from dataclasses import dataclass

from .serialize import Vector


@dataclass(frozen=True)
class Episode:
    """Эпизод памяти — прецедент правки или ситуации.

    id=None для новых эпизодов (до записи в БД).
    После recall — id заполнен из БД.

    Attributes:
        content: Текстовое содержание эпизода (правка, ситуация).
        embedding: Вектор эмбеддинга content.
        timestamp: Unix timestamp (time.time()) момента эпизода.
        valence: Валентность в момент эпизода (из energy).
        stress: Аллостатический стресс в момент эпизода.
        free_energy: F(t) в момент эпизода.
        id: Назначается БД при insert; None для новых.
    """

    content: str
    embedding: Vector
    timestamp: float
    valence: float
    stress: float
    free_energy: float
    id: int | None = None
