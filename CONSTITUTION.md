# CONSTITUTION — Правила проекта neuro2

## 1. Стиль кода

### 1.1 Python и зависимости
- Python ≥ 3.11 (версия из `.python-version`)
- Все зависимости только через **uv** (`pyproject.toml` + `uv.lock`)
- Запрещено: `pip install` вручную, `requirements.txt`

### 1.2 Форматирование и линтинг
- **ruff** — единый инструмент для format + lint
- Конфиг в `pyproject.toml`
- Обязательный вызов перед коммитом: `ruff check . && ruff format .`
- Максимальная длина строки: 88 символов

### 1.3 Type hints
- **Обязательны** на всех публичных функциях и методах
- Для `src/core/` — strict-режим (mypy или pyright)
- Запрещено: `Any`, `object` (если можно заменить на конкретный тип)
- Использовать `typing.TypedDict` для структурированных данных

### 1.4 Docstrings
- **Google-style** для всех классов и публичных функций
- Обязательные секции: `Args`, `Returns`, `Raises`
- Пример:
```python
def compute_free_energy(state: np.ndarray, precision: float) -> float:
    """Calculate free energy F(t) for the current state.
    
    Args:
        state: Current cortical state vector x(t).
        precision: Precision weighting γ for the current channel.
        
    Returns:
        Free energy value F(t) ≥ 0.
        
    Raises:
        ValueError: If state vector is empty.
    """
```

### 1.5 Именование
- Функции/переменные: `snake_case`
- Классы: `PascalCase`
- Константы: `UPPER_SNAKE_CASE`
- Приватные: `_leading_underscore`
- Абсолютные импорты: `from src.core.cmc import Column`

### 1.6 Структура функций
- Максимальная длина функции — ориентир **40 строк**
- При превышении — обязательный рефакторинг на подфункции
- Одна функция — одна ответственность

## 2. Архитектурные ограничения

### 2.1 Колоночная ткань
- Никакого синхронного блокирующего backprop через все колонки сразу
- Только локальные обновления per-column
- Параллельный батчинг: `[N_columns, In_Dim, State_Dim]`

### 2.2 Параметризуемость
- Все временные величины — параметризуемые, не захардкоженные:
  - `dt` (шаг интегрирования)
  - `TTL` задач
  - `cooldown` между вызовами
  - Пороги `F(t)`, `precision`
- Все пороги — в `src/config/`, не в коде логики

### 2.3 Внешние API
- Любое обращение к Embeddings/LLM — только через **event-triggered** вызов
- Порог `F(t)` — не фиксирован в Фазе 1, логируется для калибровки
- Не на каждый тик, а только при условии `F(t) > threshold`

### 2.4 Тестируемость
- Каждый модуль в `src/core/`, `src/memory/`, `src/tm/` обязан быть тестируем в изоляции
- Без глобального состояния
- Без синглтонов
- Зависимости — через явную инъекцию в конструктор

## 3. Тестирование

### 3.1 pytest
- Минимум **1 unit-тест** на каждую публичную функцию в `src/core/energy` и `src/core/cmc`
- Coverage threshold: **≥ 80%** для `src/core/` перед мержем
- Файлы тестов: `tests/test_<module>.py`

### 3.2 SQLite в тестах
- **Юнит-тесты**: `:memory:` — быстро, просто
- **Интеграционные**: временный файл через `tmp_path` — проверка персистентности
- **Прода**: файл на диске (`host_memory.db`)

## 4. Коммиты и PR

### 4.1 Conventional Commits
- `feat:` — новая функциональность
- `fix:` — исправление бага
- `refactor:` — рефакторинг без изменения поведения
- `docs:` — изменения в документации
- `test:` — добавление/изменение тестов
- `chore:` — вспомогательные изменения

### 4.2 PR-процесс
- Один PR на модуль (не на всю Фазу)
- Прогон `pytest` и `ruff check` в CI перед мержем
- Ручной review от человека перед мержем

## 5. Spec-Driven Development

### 5.1 Порядок
1. **Constitution** ← ты здесь
2. **Specify** — SPEC.md (без кода, только интерфейс)
3. **Plan** — PLAN.md (файлы, зависимости, порядок)
4. **Tasks** — BACKLOG.md (атомарные задачи)
5. **Implement** — код + тесты

### 5.2 Не пропускать шаги
- Любой код без SPEC.md/PLAN.md — отклонить в review
- Open Questions в конце SPEC.md — не додумывать значения по умолчанию
- Один одобренный SPEC → один PLAN → реализация

## 6. ADR (Architecture Decision Records)

### 6.1 Правила
- Папка `adr/`, нумерация `0001-`, `0002-`, ...
- **Append-only**: не редактировать задним числом
- Формат: Problem → Options → Decision → Tradeoffs → Confidence
- Новые решения — новые файлы, не правка старых
