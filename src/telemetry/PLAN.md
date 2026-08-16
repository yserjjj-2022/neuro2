# PLAN.md — src/telemetry

## Файлы для создания/изменения

1. `src/telemetry/models.py` — `TelemetryEvent` dataclass
2. `src/telemetry/writer.py` — `TelemetryWriter` (pure logic)
3. `src/telemetry/logger.py` — `TelemetryLogger` (shell + DI)
4. `src/telemetry/__init__.py` — Re-exports (обновить)
5. `src/tests/test_telemetry_writer.py` — Unit-тесты для writer
6. `src/tests/test_telemetry_logger.py` — Unit-тесты для logger

## Зависимости

**Внешние:**
- `json` — стандартная библиотека для JSONL

**Стандартная библиотека:**
- `dataclasses` — frozen dataclass для события
- `typing` — Optional, Callable
- `pathlib` — Path для log_path
- `time` — time.time() для timestamp

## Порядок реализации

### 1. Models (`models.py`)
- Создать `@dataclass(frozen=True) TelemetryEvent`
- Поля: `timestamp`, `free_energy`, `valence`, `allostatic_stress`, `active_columns`, `phase`, `mode`
- Без `would_trigger` (отложен до Фазы 2)

### 2. Writer (`writer.py`)
- Создать `TelemetryWriter`
- `__init__`: `log_path: Path`
- `write(event)`:
  1. Сериализовать event в JSON через `dataclasses.asdict()`
  2. Добавить timestamp (если не задан)
  3. Записать одну JSONL-строку в файл
- `close()`: закрыть файл

### 3. Тесты для Writer (`src/tests/test_telemetry_writer.py`)
- `test_write_creates_file`: writer создаёт файл при первом вызове
- `test_write_jsonl_format`: каждая строка — валидный JSON
- `test_write_multiple_events`: несколько событий в одном файле

### 4. Logger (`logger.py`)
- Создать `TelemetryLogger`
- `__init__`: `writer: TelemetryWriter`
- `log(event)`: вызывает `writer.write(event)`

### 5. Тесты для Logger (`src/tests/test_telemetry_logger.py`)
- `test_logger_no_writer`: logger работает без writer (mock)

### 6. `__init__.py`
- Re-export: `TelemetryEvent`, `TelemetryWriter`, `TelemetryLogger`

## План тестов

| Тест | Покрытие | Инвариант |
|------|----------|-----------|
| `test_write_creates_file` | Создание файла | Запись не блокирует |
| `test_write_jsonl_format` | JSONL формат | Одна строка = один JSON |
| `test_write_multiple_events` | Несколько событий | Monotonic timestamps |
| `test_logger_no_writer` | Logger без writer | Не блокирует |

## Заметки для реализации

- **JSONL формат**: одна строка = один JSON-объект, без пустых строк
- **Timestamp**: `time.time()` в `write()`, если не задан в event
- **Functional Core**: `TelemetryWriter.write()` — чистая функция, тестируется без файловой системы (через `tmp_path`)
- **Reference**: `FreeEnergyResult` из `src/core/energy/` — аналогичный frozen dataclass
