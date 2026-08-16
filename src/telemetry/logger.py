"""TelemetryLogger — Shell with DI through Protocol.

Acts as a shadow observer: logs F(t) without making decisions.
Injects writer via Protocol for testability.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from .models import TelemetryEvent

logger = logging.getLogger(__name__)


class SupportsWrite(Protocol):
    """Protocol для duck typing writer-а."""

    def write(self, event: TelemetryEvent) -> None: ...


class TelemetryLogger:
    """Shadow-наблюдатель: логирует F(t) без принятия решений.

    Соответствует паттерну energy:
    - Ядро (serialize_event) — чистая функция, тестируется без I/O
    - Shell (TelemetryWriter) — владеет файлом
    - Logger — инъекция writer через Protocol, можно мокать в тестах

    phase/mode задаются в __init__, подставляются при построении TelemetryEvent.

    Crash-safety: не пробрасывает исключения от writer — не должна ронять
    основной цикл хоста.

    Attributes:
        writer: Writer (любой объект с методом write(event)).
        phase: Текущая фаза проекта (из config).
        mode: Текущий режим (game/cooperative/free).
    """

    def __init__(
        self,
        writer: SupportsWrite,
        phase: str = "phase1",
        mode: str = "free",
    ) -> None:
        """Инициализация логгера.

        Args:
            writer: Writer (любой объект с методом write(event)).
            phase: Текущая фаза проекта (из config).
            mode: Текущий режим (game/cooperative/free).
        """
        self.writer = writer
        self.phase = phase
        self.mode = mode

    def log(
        self,
        free_energy: float,
        valence: float,
        allostatic_stress: float,
        active_columns: int = 0,
    ) -> None:
        """Записать событие в лог.

        Автоматически добавляет timestamp, phase, mode.
        Не пробрасывает исключения от writer.

        Args:
            free_energy: Значение F(t).
            valence: Валентность.
            allostatic_stress: Аллостатический стресс.
            active_columns: Количество активных колонок.
        """
        event = TelemetryEvent(
            timestamp=time.time(),
            free_energy=free_energy,
            valence=valence,
            allostatic_stress=allostatic_stress,
            active_columns=active_columns,
            phase=self.phase,
            mode=self.mode,
        )
        try:
            self.writer.write(event)
        except Exception:  # noqa: BLE001 — crash-safety: не роняем основной цикл
            logger.error(
                "Telemetry write failed — continuing without log entry: "
                "free_energy=%.4f, valence=%.4f, stress=%.4f, columns=%d",
                free_energy,
                valence,
                allostatic_stress,
                active_columns,
            )
