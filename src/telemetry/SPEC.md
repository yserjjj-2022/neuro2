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
    """
    timestamp: float                  # Unix timestamp (time.time())
    free_energy: float                # Сырое значение F(t)
    valence: float                    # Валентность (-dF/dt)
    allostatic_stress: float          # Интеграл F(t) по времени
    active_columns: int               # Количество активных колонок
    phase: str                        # Текущая фаза проекта (из config)
    mode: str                         # Текущий режим (game/cooperative/free)
```

### TelemetryWriter

```python
class TelemetryWriter:
    """Запись TelemetryEvent в JSONL-файл.
    
    Functional Core: write() — чистая функция (без состояния).
    Imperative Shell: path хранится в __init__.
    """
    
    def __init__(self, log_path: Path) -> None:
        """Инициализация writer.
        
        Args:
            log_path: Путь к JSONL-файлу для записи.
        """
        ...
    
    def write(self, event: TelemetryEvent) -> None:
        """Записать одно событие в JSONL.
        
        Args:
            event: Событие телеметрии для записи.
        """
        ...
    
    def close(self) -> None:
        """Закрыть файл (опционально, для cleanup)."""
        ...
```

### TelemetryLogger (DI через sink)

```python
class TelemetryLogger:
    """Shadow-наблюдатель: логирует F(t) без принятия решений.
    
    Соответствует паттерну из energy:
    - Core (TelemetryWriter) — чистая функция, тестируется без I/O
    - Shell (TelemetryLogger) — инъекция writer, можно мокать в тестах
    - В проде: writer = TelemetryWriter(Path("host.log"))
    - В тестах: writer = MagicMock()
    """
    
    def __init__(
        self,
        writer: TelemetryWriter,
    ) -> None:
        """
        Args:
            writer: Writer для записи событий.
        """
        ...
    
    def log(self, event: TelemetryEvent) -> None:
        """Записать событие в лог.
        
        Args:
            event: Событие для логирования.
        """
        ...
```

## Инварианты

1. **Плоская структура**: `TelemetryEvent` содержит только примитивные типы (float, int, str). Никакой вложенности.
2. **Неблокирующая запись**: `write()` не должен блокировать основной цикл дольше 5 мс.
3. **F(t) ≥ 0**: значение free_energy всегда неотрицательное.
4. **Monotonic timestamps**: `timestamp` каждого следующего события ≥ предыдущего.
5. **JSONL-формат**: одна строка = один JSON-объект. Пустые строки не допускаются.
6. **Функциональное ядро**: `TelemetryWriter.write()` — чистая функция, тестируется без файловой системы (через mock).

## Критерии приёмки

- [ ] `TelemetryEvent` — frozen dataclass, все поля типизированы
- [ ] `TelemetryWriter.write()` сериализует event в JSONL и записывает в файл
- [ ] `TelemetryLogger` тестируется без файловой системы (mock writer)
- [ ] Минимум 3 unit-теста: 2 для `TelemetryWriter`, 1 для `TelemetryLogger`
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
| Формат timestamp | Решено | Unix float (`time.time()`), UTC |
| Rotation логов | Вне скоупа | Фаза 4+: active pruning старых записей |
| Конфиденциальность | Вне скоупа | Фаза 3+: anonymization PII из логов |
| `would_trigger` в логе | Отложен до Фазы 2 | Калибровка порога — Фаза 2 |
