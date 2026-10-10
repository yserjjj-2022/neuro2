# SPEC.md — src/telemetry

## Назначение

Плоское JSONL-логирование состояния хоста для наблюдаемости и калибровки.
Functional Core / Imperative Shell (ADR-0004).

S1: событие расширено до 15 полей — сопоставление с тиком, теги каналов,
reflex, ресурсы, drift.
S2: +3 поля памяти (prior/hit/stored) = 18.
S3: +`spoke` = 19.
S4: +4 поля (throttle/homeostasis/policy_action/policy_reason) = 23.
S4-долг: +`escape_hatch` (право сообщить о перегрузке) = 24.
S5: +5 социальных полей (partner_trust/uncertainty/name, pause_s,
claim_conflict) = 29.
S6: +6 полей автономии (metacog_conflict/metastability/saturation, reset_level,
change_kind, consolidated_pruned) = 35.
S6 (проход 2): +2 поля MCP-зондирования (probe_affordance, probe_success) = 37.
S8: +5 полей секвенирования актуаций (actuation_status/goal/impatience/steps/
preemptions) = 42.

## Публичный интерфейс

### TelemetryEvent (frozen dataclass, 42 полей)

```python
@dataclass(frozen=True)
class TelemetryEvent:
    timestamp: float  # time.time()
    tick: int  # номер тика (S1)
    free_energy: float
    valence: float
    allostatic_stress: float
    gamma: float  # агрегат precision (S1)
    active_columns: int
    active_tags: str  # CSV активных каналов шины (S1)
    reflex_tags: str  # CSV критических сигналов (S1)
    bus_dim: int  # ширина шины (S1)
    latency_ms: float  # длительность тика, мс (S1)
    rss_mb: float  # RSS процесса, МБ (S1)
    drift: bool  # флаг детектора дрейфа (S1)
    memory_prior: float  # cos извлечённого эпизода (S2)
    memory_hit: bool  # recall нашёл эпизод (S2)
    episode_stored: bool  # эпизод записан на тике (S2)
    spoke: bool  # хост сгенерировал реплику (S3)
    throttle: bool  # активен ли рефлекс-throttle (S4)
    homeostasis: float  # max_deviation гомеостаза (S4)
    policy_action: str  # выбранное действие policy (S4)
    policy_reason: str  # причинная трассировка решения (S4)
    escape_hatch: bool  # право сообщить о перегрузке под throttle (S4)
    partner_trust: float  # доверие к партнёру, [0, 1] (S5)
    partner_uncertainty: float  # неопределённость идентичности (S5)
    partner_name: str  # принятое имя партнёра, "" если нет (S5)
    pause_s: float  # интервал с прошлой реплики, с (S5)
    claim_conflict: float  # рассогласование последнего утверждения (S5)
    metacog_conflict: float  # несогласие ансамбля колонок (S6)
    metacog_metastability: float  # частота смен аттрактора (S6)
    metacog_saturation: float  # насыщение/тренд F (S6)
    reset_level: str  # уровень сброса: ""/soft/freeze/hard (S6)
    change_kind: str  # классификация: ""/stable/development/drift (S6)
    consolidated_pruned: int  # удалено эпизодов при консолидации (S6)
    probe_affordance: str  # имя выполненного MCP-зондирования, "" если нет (S6)
    probe_success: bool  # успешно ли зондирование (S6)
    actuation_status: str  # статус BT-дерева, "" если выключено (S8)
    actuation_goal: str  # бегущая активация, "" если ничего (S8)
    actuation_impatience: float  # сигнал нетерпения [0, 1] (S8)
    actuation_steps: int  # завершено шагов актуации на тике (S8)
    actuation_preemptions: int  # преемпций на тике (S8)
    phase: str
    mode: str
```

### serialize_event (Core)

```python
def serialize_event(event: TelemetryEvent) -> str:
    """JSON-строка, без I/O. Raises ValueError при NaN/Infinity."""
```

### TelemetryWriter (Shell)

```python
class TelemetryWriter:
    def __init__(self, log_path: Path) -> None: ...
    def write(self, event: TelemetryEvent) -> None: ...
    def close(self) -> None: ...
```

### TelemetryLogger (Shell, DI)

```python
class TelemetryLogger:
    def __init__(self, writer: SupportsWrite, phase="phase1", mode="free") -> None: ...


def log(
    self,
    free_energy,
    valence,
    allostatic_stress,
    active_columns=0,
    *,
    tick=0,
    gamma=0.0,
    active_tags="",
    reflex_tags="",
    bus_dim=0,
    latency_ms=0.0,
    rss_mb=0.0,
    drift=False,
    memory_prior=0.0,
    memory_hit=False,
    episode_stored=False,
    spoke=False,
    throttle=False,
    homeostasis=0.0,
    policy_action="",
    policy_reason="",
    escape_hatch=False,
    partner_trust=0.0,
    partner_uncertainty=0.0,
    partner_name="",
    pause_s=0.0,
    claim_conflict=0.0,
    metacog_conflict=0.0,
    metacog_metastability=0.0,
    metacog_saturation=0.0,
    reset_level="",
    change_kind="",
    consolidated_pruned=0,
    probe_affordance="",
    probe_success=False,
    actuation_status="",
    actuation_goal="",
    actuation_impatience=0.0,
    actuation_steps=0,
    actuation_preemptions=0,
) -> None: ...
```

Новые поля — keyword-only с дефолтами (обратная совместимость).

## Инварианты

1. **Плоская структура**: только примитивы.
2. **Валидный JSON**: `allow_nan=False`.
3. **Crash-safety**: `log()` не пробрасывает исключения от writer.
4. **Flush после записи** (crash-safety).
5. **DI через Protocol** `SupportsWrite`.

## Критерии приёмки (S1–S6)

- [x] `TelemetryEvent` — 35 полей, все типизированы
- [x] `serialize_event` — чистая, NaN → ValueError
- [x] новые поля в JSON (проверено тестом)
- [x] обратная совместимость `log()` (дефолты)
- [x] crash-safety сохранена

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Формат `active_tags` | Решено | CSV-строка (плоский JSON) |
| Rotation логов | Вне скоупа | Фаза 4+ |
| `time_scale` в логе | Отложено | добавить при необходимости (ADR-0006) |
