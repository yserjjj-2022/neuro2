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
        *,
        tick: int = 0,
        gamma: float = 0.0,
        active_tags: str = "",
        reflex_tags: str = "",
        bus_dim: int = 0,
        latency_ms: float = 0.0,
        rss_mb: float = 0.0,
        drift: bool = False,
        memory_prior: float = 0.0,
        memory_hit: bool = False,
        episode_stored: bool = False,
        spoke: bool = False,
        throttle: bool = False,
        homeostasis: float = 0.0,
        policy_action: str = "",
        policy_reason: str = "",
    ) -> None:
        """Записать событие в лог.

        Автоматически добавляет timestamp, phase, mode.
        Не пробрасывает исключения от writer.

        Args:
            free_energy: Значение F(t).
            valence: Валентность.
            allostatic_stress: Аллостатический стресс.
            active_columns: Количество активных колонок.
            tick: Номер тика хоста.
            gamma: Precision weighting γ.
            active_tags: CSV-теги активных каналов шины.
            reflex_tags: CSV-теги критических сигналов.
            bus_dim: Ширина шины.
            latency_ms: Длительность тика, мс.
            rss_mb: RSS процесса, МБ.
            drift: Флаг детектора дрейфа.
            memory_prior: Косинус извлечённого эпизода (S2).
            memory_hit: Recall нашёл релевантный эпизод (S2).
            episode_stored: Эпизод записан на этом тике (S2).
            spoke: Хост сгенерировал реплику на тике (S3).
            throttle: Активен ли рефлекс-throttle на тике (S4).
            homeostasis: Максимальное отклонение гомеостаза (S4).
            policy_action: Выбранное policy действие ("" если не вызывалась) (S4).
            policy_reason: Причина выбора policy — трассировка (S4).
        """
        event = TelemetryEvent(
            timestamp=time.time(),
            tick=tick,
            free_energy=free_energy,
            valence=valence,
            allostatic_stress=allostatic_stress,
            gamma=gamma,
            active_columns=active_columns,
            active_tags=active_tags,
            reflex_tags=reflex_tags,
            bus_dim=bus_dim,
            latency_ms=latency_ms,
            rss_mb=rss_mb,
            drift=drift,
            memory_prior=memory_prior,
            memory_hit=memory_hit,
            episode_stored=episode_stored,
            spoke=spoke,
            throttle=throttle,
            homeostasis=homeostasis,
            policy_action=policy_action,
            policy_reason=policy_reason,
            phase=self.phase,
            mode=self.mode,
        )
        try:
            self.writer.write(event)
        except Exception:  # noqa: BLE001 — crash-safety: не роняем основной цикл
            logger.error(
                "Telemetry write failed — continuing without log entry: "
                "tick=%d, free_energy=%.4f, valence=%.4f, stress=%.4f, columns=%d",
                tick,
                free_energy,
                valence,
                allostatic_stress,
                active_columns,
            )
