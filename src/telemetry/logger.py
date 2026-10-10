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
        escape_hatch: bool = False,
        partner_trust: float = 0.0,
        partner_uncertainty: float = 0.0,
        partner_name: str = "",
        pause_s: float = 0.0,
        claim_conflict: float = 0.0,
        metacog_conflict: float = 0.0,
        metacog_metastability: float = 0.0,
        metacog_saturation: float = 0.0,
        reset_level: str = "",
        change_kind: str = "",
        consolidated_pruned: int = 0,
        probe_affordance: str = "",
        probe_success: bool = False,
        actuation_status: str = "",
        actuation_goal: str = "",
        actuation_impatience: float = 0.0,
        actuation_steps: int = 0,
        actuation_preemptions: int = 0,
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
            escape_hatch: Право сообщить о перегрузке под удержанным throttle (S4).
            partner_trust: Доверие к партнёру (S5).
            partner_uncertainty: Неопределённость идентичности партнёра (S5).
            partner_name: Принятое имя партнёра (S5).
            pause_s: Интервал с прошлой реплики, с (S5).
            claim_conflict: Рассогласование последнего утверждения (S5).
            metacog_conflict: Несогласие ансамбля колонок (S6).
            metacog_metastability: Частота смен аттрактора в окне (S6).
            metacog_saturation: Насыщение/тренд F (S6).
            reset_level: Уровень сброса: "", soft/freeze/hard (S6).
            change_kind: Классификация: ""/stable/development/drift (S6).
            consolidated_pruned: Удалено эпизодов при консолидации (S6).
            probe_affordance: Выполненное MCP-зондирование ("" если нет) (S6).
            probe_success: Успешно ли зондирование (S6).
            actuation_status: Статус BT-дерева ("" если выключено) (S8).
            actuation_goal: Бегущая активация ("" если ничего не бежит) (S8).
            actuation_impatience: Сигнал нетерпения [0, 1] (S8).
            actuation_steps: Завершено шагов актуации на тике (S8).
            actuation_preemptions: Преемпций на тике (S8).
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
            escape_hatch=escape_hatch,
            partner_trust=partner_trust,
            partner_uncertainty=partner_uncertainty,
            partner_name=partner_name,
            pause_s=pause_s,
            claim_conflict=claim_conflict,
            metacog_conflict=metacog_conflict,
            metacog_metastability=metacog_metastability,
            metacog_saturation=metacog_saturation,
            reset_level=reset_level,
            change_kind=change_kind,
            consolidated_pruned=consolidated_pruned,
            probe_affordance=probe_affordance,
            probe_success=probe_success,
            actuation_status=actuation_status,
            actuation_goal=actuation_goal,
            actuation_impatience=actuation_impatience,
            actuation_steps=actuation_steps,
            actuation_preemptions=actuation_preemptions,
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
