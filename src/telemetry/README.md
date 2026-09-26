# Telemetry

JSONL-логирование состояния хоста: F(t), valence, allostatic_stress,
active_columns + phase/mode.

Архитектура (ADR-0004):
- Чистое ядро: `serialize_event` — JSON без I/O, `allow_nan=False`
- Shell: `TelemetryWriter` — владеет файлом, flush после каждой записи
- Logger: DI через Protocol `SupportsWrite`, crash-safety (не роняет loop)

Формат: одна строка = один JSON-объект (JSONL).
Назначение: наблюдаемость и калибровка порогов (Фаза 2).

Планируется расширение полей: `tick`, `active_tags`, `reflex_tags`, `bus_dim`.