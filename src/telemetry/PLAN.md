# PLAN.md — src/telemetry

## Файлы для создания/изменения

1. `src/telemetry/models.py` — `TelemetryEvent` dataclass
2. `src/telemetry/serialize.py` — `serialize_event()` (чистое ядро)
3. `src/telemetry/writer.py` — `TelemetryWriter` (shell + I/O)
4. `src/telemetry/logger.py` — `TelemetryLogger` (DI через Protocol)
5. `src/telemetry/__init__.py` — Re-exports (обновить)
6. `src/tests/test_telemetry_serialize.py` — Unit-тесты для serialize_event
7. `src/tests/test_telemetry_writer.py` — Unit-тесты для writer
8. `src/tests/test_telemetry_logger.py` — Unit-тесты для logger

## Зависимости

**Внешние:**
- Нет

**Стандартная библиотека:**
- `dataclasses` — frozen dataclass для события
- `typing` — Protocol для duck typing
- `pathlib` — Path для log_path
- `time` — time.time() для timestamp
- `json` — json.dumps для сериализации
- `logging` — для crash-safety логирования

## Порядок реализации

### 1. Models (`models.py`)
- Создать `@dataclass(frozen=True) TelemetryEvent`
- Поля: `timestamp`, `free_energy`, `valence`, `allostatic_stress`, `active_columns`, `phase`, `mode`
- phase/mode заполняются TelemetryLogger.log(), не caller-ом напрямую

### 2. Serialize (`serialize.py`) — Functional Core
- Создать `serialize_event(event: TelemetryEvent) -> str`
- Использовать `json.dumps(dataclasses.asdict(event), allow_nan=False)`
- Возвращает JSON-строку, без I/O
- Raises `ValueError` при NaN/Infinity

### 3. Тесты для Serialize (`src/tests/test_telemetry_serialize.py`)
- `test_serialize_valid`: валидный event → корректная JSON-строка
- `test_serialize_nan_raises`: NaN → ValueError

### 4. Writer (`writer.py`) — Imperative Shell
- Создать `TelemetryWriter`
- `__init__`: `log_path: Path`, открыть файл один раз
- `write(event)`:
  1. Вызывает `serialize_event(event)` для получения строки
  2. Записывает строку + `\n` в файл
  3. Вызывает `self._file.flush()` (crash-safety)
- `close()`: закрыть файл

### 5. Тесты для Writer (`src/tests/test_telemetry_writer.py`)
- `test_write_creates_file`: writer создаёт файл при первом вызове
- `test_write_jsonl_format`: каждая строка — валидный JSON

### 6. Logger (`logger.py`) — Shell с DI
- Создать Protocol `SupportsWrite`
- Создать `TelemetryLogger`
- `__init__`: `writer: SupportsWrite`, `phase: str`, `mode: str`
- `log(free_energy, valence, stress, active_columns)`:
  1. Создаёт `TelemetryEvent` с `time.time()`, `phase`, `mode`
  2. Вызывает `writer.write(event)` в try/except
  3. При ошибке — `logging.error()`, не пробрасывает дальше
- `__init__`: `writer: SupportsWrite | None = None` (для тестов без writer)

### 7. Тесты для Logger (`src/tests/test_telemetry_logger.py`)
- `test_logger_with_mock_writer`: logger с mock writer (проверяет вызов)
- `test_logger_swallows_writer_errors`: logger не пробрасывает исключения от writer

### 8. `__init__.py`
- Re-export: `TelemetryEvent`, `serialize_event`, `TelemetryWriter`, `TelemetryLogger`

## План тестов

| Тест | Покрытие | Инвариант |
|------|----------|-----------|
| `test_serialize_valid` | Сериализация | JSON-строка из event |
| `test_serialize_nan_raises` | Валидация | ValueError при NaN |
| `test_write_creates_file` | Создание файла | Запись не блокирует |
| `test_write_jsonl_format` | JSONL формат | Одна строка = один JSON |
| `test_logger_with_mock_writer` | DI через Protocol | Не блокирует |
| `test_logger_swallows_writer_errors` | Crash-safety | Не роняет основной цикл |

## Заметки для реализации

- **JSONL формат**: одна строка = один JSON-объект, без пустых строк
- **Timestamp**: `time.time()` в `logger.log()`, не в serialize_event
- **Functional Core**: `serialize_event()` — чистая функция, тестируется без файловой системы
- **Imperative Shell**: `TelemetryWriter` — единственное место I/O
- **DI через Protocol**: `TelemetryLogger` зависит от `SupportsWrite`, не от конкретного `TelemetryWriter`
- **phase/mode**: задаются в `TelemetryLogger.__init__`, подставляются автоматически при построении TelemetryEvent
- **Crash-safety**: `logger.log()` — try/except, logging.error(), не пробрасывает дальше
- **Flush-политика**: `writer.write()` → `self._file.flush()` после каждой записи
- **Reference**: `FreeEnergyResult` из `src/core/energy/` — аналогичный frozen dataclass
