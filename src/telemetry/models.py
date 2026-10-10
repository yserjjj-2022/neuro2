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
        memory_prior: Косинус извлечённого эпизода (0.0 если нет) — S2.
        memory_hit: Recall нашёл релевантный эпизод — S2.
        episode_stored: Эпизод записан на этом тике — S2.
        spoke: Хост сгенерировал реплику на тике — S3.
        throttle: Активен ли рефлекс-throttle на тике — S4.
        homeostasis: Максимальное отклонение гомеостаза — S4.
        policy_action: Выбранное policy действие ("" если не вызывалась) — S4.
        policy_reason: Причина выбора policy (трассировка) — S4.
        escape_hatch: Право сообщить о перегрузке под удержанным throttle — S4.
        partner_trust: Доверие к партнёру, [0, 1] — S5.
        partner_uncertainty: Неопределённость идентичности партнёра — S5.
        partner_name: Принятое имя партнёра ("" если не объявлено) — S5.
        pause_s: Интервал с прошлой реплики, с — S5.
        claim_conflict: Рассогласование последнего утверждения — S5.
        metacog_conflict: Несогласие ансамбля колонок — S6.
        metacog_metastability: Частота смен аттрактора в окне — S6.
        metacog_saturation: Насыщение/тренд F — S6.
        reset_level: Уровень сброса ("", soft/freeze/hard) — S6.
        change_kind: Классификация изменения (""/stable/development/drift) — S6.
        consolidated_pruned: Удалено эпизодов при консолидации — S6.
        probe_affordance: Имя выполненного MCP-зондирования ("" если нет) — S6.
        probe_success: Успешно ли зондирование (gate разрешил и транспорт ответил).
        actuation_status: Статус BT-дерева ("" если выключено) — S8.
        actuation_goal: Бегущая активация ("" если ничего не бежит) — S8.
        actuation_impatience: Сигнал нетерпения [0, 1] — S8.
        actuation_steps: Завершено шагов актуации на тике — S8.
        actuation_preemptions: Преемпций (смен целей) на тике — S8.
        channel_contrib: CSV-вклады каналов шины в F (tag:вклад) — BACKLOG.
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
    memory_prior: float
    memory_hit: bool
    episode_stored: bool
    spoke: bool
    throttle: bool
    homeostasis: float
    policy_action: str
    policy_reason: str
    escape_hatch: bool
    partner_trust: float
    partner_uncertainty: float
    partner_name: str
    pause_s: float
    claim_conflict: float
    metacog_conflict: float
    metacog_metastability: float
    metacog_saturation: float
    reset_level: str
    change_kind: str
    consolidated_pruned: int
    probe_affordance: str
    probe_success: bool
    actuation_status: str
    actuation_goal: str
    actuation_impatience: float
    actuation_steps: int
    actuation_preemptions: int
    channel_contrib: str
    phase: str
    mode: str
