# SPEC.md — src/telemetry

## Назначение

Плоское JSONL-логирование состояния хоста для наблюдаемости и калибровки.
Functional Core / Imperative Shell (ADR-0004).

S1: событие расширено до 15 полей — сопоставление с тиком, теги каналов,
reflex, ресурсы, drift.

## Публичный интерфейс

### TelemetryEvent (frozen dataclass, 15 полей)

```python
@dataclass(frozen=True)
class TelemetryEvent:
    timestamp: float        # time.time()
    tick: int               # номер тика (S1)
    free_energy: float
    valence: float
    allostatic_stress: float
    gamma: float            # агрегат precision (S1)
    active_columns: int
    active_tags: str        # CSV активных каналов шины (S1)
    reflex_tags: str        # CSV критических сигналов (S1)
    bus_dim: int            # ширина шины (S1)
    latency_ms: float       # длительность тика, мс (S1)
    rss_mb: float           # RSS процесса, МБ (S1)
    drift: bool             # флаг детектора дрейфа (S1)
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
        self, free_energy, valence, allostatic_stress, active_columns=0,
        *, tick=0, gamma=0.0, active_tags="", reflex_tags="",
        bus_dim=0, latency_ms=0.0, rss_mb=0.0, drift=False,
    ) -> None: ...
```

Новые поля — keyword-only с дефолтами (обратная совместимость).

## Инварианты

1. **Плоская структура**: только примитивы.
2. **Валидный JSON**: `allow_nan=False`.
3. **Crash-safety**: `log()` не пробрасывает исключения от writer.
4. **Flush после записи** (crash-safety).
5. **DI через Protocol** `SupportsWrite`.

## Критерии приёмки (S1)

- [x] `TelemetryEvent` — 15 полей, все типизированы
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
