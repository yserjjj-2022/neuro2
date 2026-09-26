# PLAN.md — src/telemetry

## Файлы

1. `src/telemetry/models.py` — `TelemetryEvent` (15 полей) ✅
2. `src/telemetry/serialize.py` — `serialize_event` (Core) ✅
3. `src/telemetry/writer.py` — `TelemetryWriter` (Shell) ✅
4. `src/telemetry/logger.py` — `TelemetryLogger` + `SupportsWrite` ✅
5. `src/telemetry/__init__.py` — re-exports ✅
6. `src/tests/test_telemetry_serialize.py` ✅
7. `src/tests/test_telemetry_writer.py` ✅
8. `src/tests/test_telemetry_logger.py` ✅

## Зависимости

**Стандартная библиотека:** `dataclasses`, `typing`, `pathlib`, `time`, `json`, `logging`.

## Порядок реализации (S1, выполнено)

1. Расширить `TelemetryEvent` 8 новыми полями (tick, gamma, active_tags,
   reflex_tags, bus_dim, latency_ms, rss_mb, drift).
2. `TelemetryLogger.log(...)`: keyword-only параметры с дефолтами.
3. Обновить тесты (все 15 полей, обратная совместимость).

## План тестов

| Тест | Покрытие | Инвариант |
|---|---|---|
| `test_serialize_all_s1_fields` | 15 полей | присутствуют в JSON |
| `test_serialize_nan_raises` | NaN/inf | ValueError |
| `test_write_jsonl_format` | JSONL | одна строка = JSON |
| `test_logger_with_mock_writer` | новые поля | DI |
| `test_logger_swallows_writer_errors` | crash-safety | не роняет loop |

## Заметки

- **Плоская структура** сохранена (только примитивы).
- **Обратная совместимость**: новые поля keyword-only с дефолтами.
- **Loop владеет writer'ом** (S1): pipeline больше не логирует.
