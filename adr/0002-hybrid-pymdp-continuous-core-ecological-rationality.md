# ADR-0002: Гибридное ядро (pymdp + NumPy/SciPy) для экологической рациональности

## Date
2025-01-16

## Status
Accepted

## Context
Система должна поддерживать как непрерывную динамику (F(t), x(t), valence), так и дискретное макро-целеполагание (режимы, задачи). Вопрос: как реализовать оба уровня без комбинаторного взрыва состояний?

## Decision
**Гибридный подход**: pymdp — точечно для 4-8 дискретных макро-состояний, NumPy/SciPy — для всей непрерывной динамики.

### Почему не pure pymdp
- **POMDP-матрица**: полный POMDP растёт экспоненциально с числом состояний
- **Для 4 макро-состояния × 8 состояний партнёра × 5 задач = 160 состояний** — это уже серьёзная матрица
- **pymdp оптимизирован для small discrete-state POMDPs**, не для big continuous

### Почему не pure NumPy
- **Дискретные макро-режимы** (игровой аватар / со-игрок / свободный хост) — по природе дискретны
- **pymdp предоставляет**: efficient belief update, active inference для discrete actions, structured factorization
- **Реже, чем непрерывная динамика**: дискретные решения принимаются раз в десятки тиков, непрерывные — каждый тик

### Архитектура
```
Continuous layer (NumPy/SciPy) — каждый тик
  ├── x(t) динамика колонок
  ├── F(t), valence, stress
  └── precision weighting (γ)

Discrete layer (pymdp) — каждые N тиков
  ├── macro_mode ∈ {game, cooperative, free}
  ├── partner_state ∈ {agree, neutral, conflict}
  └── active_task ∈ {task1, task2, ..., idle}
```

**Разреженная факторизация**: вместо единой POMDP-матрицы [mode × partner × task] → 3 независимых маленьких фактора. Это O(N) вместо O(N³).

## Consequences
### Положительные
- Экологическая рациональность: полный расчёт только при F(t) > threshold
- O(N) масштабирование вместо O(N³)
- Каждая подсистема использует лучший инструмент для своей задачи
- Возможность A/B сравнения: pure NumPy vs hybrid для дискретных решений

### Отрицательные
- Два разных API (pymdp + NumPy) → больше кода для поддержки
- Интерфейс между continuous и discrete слоями требует тщательного проектирования
- pymdp — нишевая библиотека, меньше документации и community support

### Tradeoffs
- **Простота vs мощь**: pure NumPy проще, но не даёт active inference для дискретных решений
- **Скорость vs точность**: discretization вводит аппроксимацию, но это осознанный компромисс

## Confidence
Medium-High. Гибридный подход теоретически обоснован (манифест, раздел 3.F), но эмпирическая валидация потребует Phase 1-2.

## References
- [Манифест, раздел 3.F — Ecological Rationality](../host_architecture_manifest.md)
- [inferactively/pymdp](https://github.com/inferactively/pymdp)
- [Active Inference: The free energy principle in robotics and neuroscience](https://www.cambridge.org/core/books/active-inference/)
