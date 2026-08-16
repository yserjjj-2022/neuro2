# SPEC.md — src/telemetry

## Назначение
Модуль логирования состояния хоста: непрерывная запись F(t), valence, allostatic stress и метаданных в JSONL-формат для последующего анализа и калибровки порогов.

См. также `FreeEnergyResult` из `src/core/energy/` — аналогичный frozen dataclass для сериализации.

## Публичный интерфейс

### TelemetryEvent (dataclass)

```python
@dataclass(frozen=True)
class TelemetryEvent:
    """Событие телеметрии — плоская структура, сериализуемая в JSON.
    
    Аналог FreeEnergyResult из src/core/energy/, но для логирования.
    phase/mode заполняются TelemetryLogger.log(), а не вызывающим кодом.
    """
    timestamp: float                  # Unix timestamp (time.time())
    free_energy: float                # Сырое значение F(t)
    valence: float                    # Валентность (-dF/dt)
    allostatic_stress: float          # Интеграл F(t) по времени
    active_columns: int               # Количество активных колонок
    phase: str                        # Фаза проекта (из config)
    mode: str                         # Режим (game/cooperative/free)
```

### Serialize (Functional Core — чистая функция)

```python
def serialize_event(event: TelemetryEvent) -> str:
    """Сериализация TelemetryEvent в JSON-строку.
    
    Чистая функция: вход → выход, без I/O, без файлов.
    Тестируется без tmp_path, без mock — просто input → output.
    
    Args:
        event: Событие для сериализации.
        
    Returns:
        JSON-строка (валидный JSON по RFC 8259).
        
    Raises:
        ValueError: Если event содержит NaN/Infinity (невалидный JSON).
        
    Note:
        Используется allow_nan=False для защиты от тихой порчи данных.
    """
    ...
```

### TelemetryWriter (Imperative Shell — владеет файлом)

```python
class TelemetryWriter:
    """Imperative Shell: единственное место I/O в модуле.
    
    Делегирует сериализацию чистому ядру (serialize_event),
    управляет файлом на диске.
    
    Открывает файл один раз при __init__, закрывает при close().
    flush() после каждой записи для crash-safety.
    """
    
    def __init__(self, log_path: Path) -> None:
        """Инициализация writer.
        
        Args:
            log_path: Путь к JSONL-файлу для записи.
        """
        ...
    
    def write(self, event: TelemetryEvent) -> None:
        """Записать одно событие в JSONL.
        
        1. Вызывает serialize_event(event) для получения строки
        2. Записывает строку + "\\n" в файл
        3. Вызывает self._file.flush() для crash-safety
        
        Args:
            event: Событие для записи.
        """
        ...
    
    def close(self) -> None:
        """Закрыть файл (опционально, для cleanup)."""
        ...
```

### TelemetryLogger (Shell с DI через Protocol)

```python
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
    """
    
    def __init__(
        self,
        writer: SupportsWrite,
        phase: str = "phase1",
        mode: str = "free",
    ) -> None:
        """
        Args:
            writer: Writer (любой объект с методом write(event)).
            phase: Текущая фаза проекта (из config).
            mode: Текущий режим (game/cooperative/free).
        """
        ...
    
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
        ...
```

## Инварианты

1. **Плоская структура**: `TelemetryEvent` содержит только примитивные типы (float, int, str). Никакой вложенности.
2. **Неблокирующая запись**: `write()` не должен блокировать основной цикл дольше 5 мс.
3. **F(t) ≥ 0**: значение free_energy всегда неотрицательное.
4. **Timestamp — wall-clock**: `time.time()` для человеческого анализа; порядок событий гарантируется порядком записи в файл, а не значением timestamp.
5. **JSONL-формат**: одна строка = один JSON-объект. Пустые строки не допускаются.
6. **Валидный JSON**: `allow_nan=False` — ValueError при попытке записать NaN/Infinity.
7. **Функциональное ядро**: `serialize_event()` — чистая функция, тестируется без файловой системы.
8. **DI через Protocol**: `TelemetryLogger` зависит от `SupportsWrite`, не от конкретного `TelemetryWriter`.
9. **Crash-safety**: `TelemetryLogger.log()` не пробрасывает исключения от writer — не должна ронять основной цикл хоста. Ошибка логируется через `logging.error()`.
10. **Flush-политика**: `write()` вызывает `self._file.flush()` после каждой записи для crash-safety.

## Критерии приёмки

- [ ] `TelemetryEvent` — frozen dataclass, все поля типизированы (включая phase, mode)
- [ ] `serialize_event()` — чистая функция: input → JSON-строка, без I/O
- [ ] `serialize_event()` raises ValueError при NaN/Infinity
- [ ] `TelemetryWriter` делегирует сериализацию `serialize_event()`
- [ ] `TelemetryWriter.write()` вызывает flush() после записи
- [ ] `TelemetryLogger` задаёт phase/mode в __init__, не в каждом log()
- [ ] `TelemetryLogger.log()` не пробрасывает исключения от writer
- [ ] `TelemetryLogger` тестируется без файловой системы (mock writer)
- [ ] Минимум 6 unit-тестов: 2 для serialize_event, 2 для writer, 2 для logger
- [ ] `ruff check` и `ruff format` проходят без ошибок
- [ ] mypy strict для `src/telemetry/` не ругается

## Явно НЕ входит в скоуп

- **Фильтрация/агрегация**: нет фильтрации по времени, нет rolling-средних
- **Удаление старых логов**: нет ротации файлов, нет active pruning
- **Визуализация**: нет построения графиков, нет dashboard
- **Реактивное поведение**: нет вызова LLM, нет изменения поведения хоста
- **Структурное логирование**: нет форматирования для человека, только JSONL для машины

## Open Questions

| Вопрос | Статус | Решение |
|--------|--------|---------|
| Формат timestamp | Решено | `time.time()` (wall-clock), не монотонный. Порядок событий — по порядку записи в файл. |
| Rotation логов | Вне скоупа | Фаза 4+: active pruning старых записей |
| Конфиденциальность | Вне скоупа | Фаза 3+: anonymization PII из логов |
| `would_trigger` в логе | Отложен до Фазы 2 | Калибровка порога — Фаза 2 |
