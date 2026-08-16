# ADR-0004: Functional Core / Imperative Shell для всех модулей src/core/*

## Date
2025-01-16

## Status
Accepted

## Context
Модуль `src/core/energy` показал, что разделение на "чистую функцию" (FreeEnergyCalculator) и "shell с состоянием" (EnergyObserver) даёт:
- Легко тестируемое ядро (stateless, нет副作用)
- Предсказуемое поведение (один вход → один выход)
- Явное управление состоянием в shell (Observer хранит _prev_f, _prev_stress)

Вопрос: применять ли этот паттерн ко всем будущим модулям `src/core/*` (cmc, voting, tm)?

## Decision
**Применить Functional Core / Imperative Shell ко всем модулям в `src/core/*`.**

### Архитектура
```
src/core/
├── energy/
│   ├── calculator.py    # Functional Core: чистая функция
│   └── observer.py      # Imperative Shell: состояние + DI
├── cmc/
│   ├── column.py        # Functional Core: forward() чистый
│   └── ensemble.py      # Imperative Shell: state, aggregation
├── voting/
│   ├── kwta.py          # Functional Core: k-WTA алгоритм
│   └── manager.py       # Imperative Shell: state, tuning
```

### Правила
1. **Core**: чистые функции, no side effects, all state via parameters
2. **Shell**: mutable state, I/O, DI через constructor
3. **Core не зависит от Shell**: core можно тестировать без shell
4. **Shell зависит от Core**: shell оборачивает core

## Consequences

### Положительные
- Тестируемость: core тестируется в изоляции
- Рефакторинг: можно менять shell, не ломая core
- Документация: архитектура самоочевидна из имен файлов
- Consistency: все модули следуют одному паттерну

### Отрицательные
- Больше файлов: один модуль = 2 файла вместо одного
- Немного boilerplate: нужно явно разделять core и shell
- Научение: новый участник должен понять паттерн

### Tradeoffs
- **Простота vs модульность**: один файл проще, но hard-to-test; два файла чуть сложнее, но легко тестируется
- **Скорость vs качество**: на быструю реализацию — один файл; на production-код — разделение

## Confidence
High. Паттерн доказан в energy, применим к cmc и voting без адаптации.

## References
- [Functional Core, Imperative Shell — Gabriel Gonzalez](https://www.youtube.com/watch?v=ZQ7haFE0f6g)
- [Clean Architecture — Robert C. Martin](https://www.amazon.com/Clean-Architecture-Artificial-Software-Structure/dp/0134494164)
- [SPEC.md energy — Functional Core section](../src/core/energy/SPEC.md)
