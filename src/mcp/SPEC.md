# SPEC.md — src/mcp

## Назначение

Signal taxonomy и registry для MCP-интеграции. Явное разведение входящих сигналов на три категории (манифест §3.З), контракт `SignalSource` и `SignalRegistry` для агрегации в единый вектор `u(t)`.

## Категории сигналов

### Exteroceptive (экстероцептивные)
- **Что:** контекст внешнего мира (погода, локация, MCP-ресурсы)
- **Обработка:** проекция в общее пространство D, агрегация стандартным путём
- **Примеры:** `weather`, `location`, `news`

### Interoceptive (интероцептивные)
- **Что:** внутреннее состояние системы (батарея, CPU, гомеостатика)
- **Обработка:** гомеостатический приор с высокой точностью; отклонение → действие
- **Reflex-path:** при `severity ≥ 0.9` — мгновенный обход EMA-сглаживания
- **Примеры:** `battery`, `cpu_temp`, `memory_usage`

### Communicative (коммуникативные)
- **Что:** текст собеседника, сообщения
- **Обработка:** ожидают явного решения о политике (ответить / промолчать / инициировать)
- **Примеры:** `user_message`, `system_notification`

## Публичный интерфейс

### SignalCategory (Enum)

```python
class SignalCategory(Enum):
    EXTEROCEPTIVE = "exteroceptive"
    INTEROCEPTIVE = "interoceptive"
    COMMUNICATIVE = "communicative"
```

### SignalSource (frozen dataclass)

```python
@dataclass(frozen=True)
class SignalSource:
    category: SignalCategory      # Категория сигнала
    data: Vector                  # Вектор данных (shape зависит от категории)
    severity: float = 0.0         # 0.0–1.0, где 1.0 — критический
    is_reflex: bool = False       # Обход EMA + dwell
    tag: str = ""                 # Идентификатор источника

    def __post_init__(self) -> None:
        # severity ∈ [0.0, 1.0]
        # interoceptive с severity ≥ 0.9 → is_reflex = True
        # is_reflex=True только для interoceptive
```

### SignalRegistry (Imperative Shell)

```python
class SignalRegistry:
    def __init__(self, active_threshold: float = 0.0) -> None:
        """Создание registry с порогом активности."""
        ...

    def register(self, signal: SignalSource) -> None:
        """Добавить источник сигнала."""
        ...

    def unregister_by_tag(self, tag: str) -> bool:
        """Удалить источники по tag. Возвращает True, если удалены."""
        ...

    def clear(self) -> None:
        """Удалить все источники."""
        ...

    def get_by_category(self, category: SignalCategory) -> list[SignalSource]:
        """Фильтрация по категории."""
        ...

    def get_reflex_signals(self) -> list[SignalSource]:
        """Получить reflex-сигналы (только interoceptive severity ≥ 0.9)."""
        ...

    def aggregate(self) -> Optional[Vector]:
        """Агрегировать все сигналы в единый вектор u(t).
        
        В Фазе 1: простая конкатенация.
        В Фазе 2+: проекции по специализациям.
        
        Returns: Vector shape=(total_dim,) или None если нет источников.
        """
        ...

    @property
    def active_threshold(self) -> float:
        """Порог активности колонок."""
        ...
```

## Инварианты

1. **FC/IS:** `SignalSource` — frozen dataclass,Immutable. `SignalRegistry` — единственный владелец списка источников.
2. **Severity → Reflex:** `severity ≥ 0.9` для interoceptive автоматически устанавливает `is_reflex=True`.
3. **Reflex guard:** `is_reflex=True` для не-interoceptive сигналов → `ValueError`.
4. **Fail-fast:** `severity ∉ [0.0, 1.0]` → `ValueError`.
5. **Non-blocking:** `aggregate()` — O(N) по числу источников, без I/O.
6. **Type safety:** data — строго `np.ndarray`, не None (None → `aggregate()` возвращает None, но data внутри SignalSource не может быть None).

## Критерии приёмки

- [ ] `SignalCategory` — Enum с тремя значениями
- [ ] `SignalSource` — frozen dataclass, immutable
- [ ] `SignalSource.__post_init__` — severity validation
- [ ] `SignalSource.__post_init__` — reflex auto-set для interoceptive severity ≥ 0.9
- [ ] `SignalSource.__post_init__` — ValueError для is_reflex + non-interoceptive
- [ ] `SignalRegistry.register()` — добавление источника
- [ ] `SignalRegistry.unregister_by_tag()` — удаление, возвращает True/False
- [ ] `SignalRegistry.clear()` — очистка
- [ ] `SignalRegistry.get_by_category()` — фильтрация
- [ ] `SignalRegistry.get_reflex_signals()` — reflex-фильтрация
- [ ] `SignalRegistry.aggregate()` — конкатенация векторов
- [ ] `SignalRegistry.aggregate()` — None при пустом списке
- [ ] `ruff check` и `ruff format` проходят без ошибок
- [ ] mypy strict для `src/mcp/` не ругается

## Явно НЕ входит в скоуп (Phase 1)

- **MCP-сервер:** нет реализации MCP Transport, это только контракт данных
- **Проекции по специализациям:** aggregate() = конкатенация, в Фазе 2 — проекции
- **Reflex-path routing:** только data-контракт, routing — wiring.py (Фаза 3)
- **SignalSource production:** нет генерации сигналов, только хранение и агрегация
- **Эмбеддинг текста:** communicative signals требуют embedding, это отдельно

## Open Questions

| Вопрос | Статус | Решение |
|--------|--------|---------|
| **Как aggregate обрабатывает разные размеры векторов?** | Решено | Конкатенация по умолчанию (разные размерности допустимы). Проекции — Фаза 2. |
| **Может ли signal.data быть None?** | Решено | Нет, data — обязательное поле Vector. Пустой сигнал = SignalSource с shape=(0,) вектором. |
| **Порог severity для reflex** | Решено | 0.9 (манифест: «критический»). Настраиваемый в Future — Фаза 3. |
