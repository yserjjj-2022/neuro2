# SPEC.md — src/telemetry

## Назначение
Модуль логирования состояния хоста: непрерывная запись F(t), valence, allostatic stress и метаданных в JSONL-формат для последующего анализа и калибровки порогов.

## Публичный интерфейс

### TelemetryEvent (dataclass)
```python
@dataclass(frozen=True)
class TelemetryEvent:
    """Событие телеметрии — плоская структура, сериализуемая в JSON."""
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
    """Запись TelemetryEvent в JSONL-файл."""
    
    def __init__(self, log_path: Path) -> None:
        """Инициализация writer.
        
        Args:
            log_path: Путь к JSONL-файлу для записи.
        """
        ...
    
    def write(self, event: TelemetryEvent) -> None:
        """Записать одно событие.
        
        Args:
            event: Событие телеметрии для записи.
        """
        ...
    
    def close(self) -> None:
        """Закрыть файл (опционально, для cleanup)."""
        ...
```

### TelemetryObserver (shadow mode)
```python
class TelemetryObserver:
    """Shadow-наблюдатель: логирует F(t) без принятия решений.
    
    Работает в Фазе 1 в режиме "shadow mode":
    - Считает, где сработал бы порог
    - Логирует, но ничего не вызывает
    - Ничего не блокирует
    """
    
    def __init__(
        self,
        writer: TelemetryWriter,
        f_threshold: Optional[float] = None,
    ) -> None:
        """
        Args:
            writer: Writer для записи событий.
            f_threshold: Порог F(t) для анализа (None = shadow mode, 
                         только логирует без триггера).
        """
        ...
    
    def observe(self, event: TelemetryEvent) -> None:
        """Наблюдать за событием и логировать.
        
        В shadow mode (f_threshold=None): логирует всегда.
        В calibrated mode: логирует и помечает, где бы сработал триггер.
        
        Args:
            event: Событие для наблюдения.
        """
        ...
```

## Инварианты

1. **Плоская структура**: `TelemetryEvent` содержит только примитивные типы (float, int, str). Никакой вложенности.
2. **Неблокирующая запись**: `write()` не должен блокировать основной цикл дольше 5 мс.
3. **F(t) ≥ 0**: значение free_energy всегда неотрицательное.
4. **Monotonic timestamps**: `timestamp` каждого следующего события ≥ предыдущего.
5. **JSONL-формат**: одна строка = один JSON-объект. Пустые строки не допускаются.

## Критерии приёмки

- [ ] `TelemetryEvent` — frozen dataclass, все поля типизированы
- [ ] `TelemetryWriter.write()` сериализует event в JSONL и записывает в файл
- [ ] `TelemetryObserver.observe()` логирует событие без блокировок
- [ ] Shadow mode работает: `f_threshold=None` → логирует всё, не триггерит действий
- [ ] Минимум 2 unit-теста: один для `TelemetryWriter`, один для `TelemetryObserver`
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
