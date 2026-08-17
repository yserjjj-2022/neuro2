# PLAN.md — src/mcp

## Файлы для создания/изменения

1. `src/mcp/models.py` — `SignalCategory` enum + `SignalSource` frozen dataclass ✅
2. `src/mcp/registry.py` — `SignalRegistry` (shell) ✅
3. `src/mcp/__init__.py` — Re-exports ✅
4. `src/mcp/SPEC.md` — Спецификация ✅
5. `src/tests/test_mcp_models.py` — Unit-тесты для моделей
6. `src/tests/test_mcp_registry.py` — Unit-тесты для registry

## Зависимости

**Внешние:**
- `numpy` — векторные операции

**Стандартная библиотека:**
- `dataclasses` — frozen dataclass
- `enum` — Enum для категорий
- `typing` — Optional, Any
- `logging` — debug-логи

## Порядок реализации

### 1. Тесты для моделей (`src/tests/test_mcp_models.py`)

- `test_signal_category_enum`: все три значения существуют
- `test_signal_source_frozen`: попытка мутации → FrozenInstanceError
- `test_signal_source_valid`: валидный сигнал создаётся
- `test_signal_source_severity_too_low`: severity < 0 → ValueError
- `test_signal_source_severity_too_high`: severity > 1 → ValueError
- `test_signal_source_reflex_auto_set`: interoceptive severity=0.95 → is_reflex=True
- `test_signal_source_reflex_non_interoceptive`: exteroceptive is_reflex=True → ValueError

### 2. Тесты для Registry (`src/tests/test_mcp_registry.py`)

- `test_registry_empty`: пустой registry → aggregate() = None
- `test_registry_register`: register → count == 1
- `test_registry_unregister_by_tag`: unregister → count == 0, возвращает True
- `test_registry_unregister_nonexistent`: unregister → возвращает False
- `test_registry_clear`: clear → count == 0
- `test_registry_get_by_category`: фильтрация по категории
- `test_registry_get_reflex_signals`: reflex-фильтрация
- `test_registry_aggregate`: конкатенация векторов
- `test_registry_aggregate_multiple`: несколько сигналов → объединённый вектор

### 3. `__init__.py` — Re-exports ✅

## План тестов

| Тест | Покрытие | Инвариант |
|------|----------|-----------|
| `test_signal_category_enum` | Enum | 3 значения |
| `test_signal_source_frozen` | Immutability | FrozenInstanceError |
| `test_signal_source_valid` | Создание | Без исключений |
| `test_signal_source_severity_too_low` | Валидация | ValueError |
| `test_signal_source_severity_too_high` | Валидация | ValueError |
| `test_signal_source_reflex_auto_set` | Reflex auto | is_reflex=True |
| `test_signal_source_reflex_non_interoceptive` | Reflex guard | ValueError |
| `test_registry_empty` | Edge case | aggregate() = None |
| `test_registry_register` | Базовый | count == 1 |
| `test_registry_unregister_by_tag` | Удаление | count == 0, True |
| `test_registry_unregister_nonexistent` | Удаление | False |
| `test_registry_clear` | Очистка | count == 0 |
| `test_registry_get_by_category` | Фильтрация | Правильный список |
| `test_registry_get_reflex_signals` | Reflex | Только reflex |
| `test_registry_aggregate` | Агрегация | Конкатенация |
| `test_registry_aggregate_multiple` | Агрегация | Shape суммарный |

## Заметки для реализации

- **Frozen dataclass**: `SignalSource` immutable, аналог FreeEnergyResult
- **Severity → Reflex**: автоматическое установление is_reflex — ключевое архитектурное свойство
- **Reflex guard**: защита от misuse — ValueError при is_reflex для не-interoceptive
- **Aggregate**: в Фазе 1 — простая конкатенация, в Фазе 2 — проекции по специализациям
- **Type hints**: Vector — numpy generic (идиома из memory/serialize.py)
