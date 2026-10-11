# ADR-0014: Целевой цикл — `selected_goal`, явные исходы, выбор цели из наблюдаемого состояния

## Status
Accepted

## Date
2026-10-11

## Context

Целевой цикл (goal loop) — отсутствующая часть архитектуры: `affordance`-скоринг
(ADR-0012) оценивает, что можно сделать; локальный мир (ADR-0013) наблюдает,
что есть. Между ними нет связки, которая **наблюдает состояние, определяет
неудовлетворённый исход, выбирает цель, исполняет и проверяет результат новым
наблюдением**.

Существующий S6-fallback (`select_affordance` в `host/loop.py`) не меняются в
этом разрезе. Целевой цикл подключается отдельно (ADR-0012 §4).

## Decision

### 1. Разделение `GoalSpec` и `OutcomeSpec`

`GoalSpec` не содержит `predicate_id` напрямую — он ссылается на `OutcomeSpec`
через `outcome_id`. `OutcomeSpec` ссылается на предикат через `predicate_id`.
Это позволяет нескольким целям разделять один критерий проверки и менять
критерий без изменения цели.

### 2. `selected_goal` — явное поле

`selected_goal` — результат функции `select_goal()`, не побочный результат
скоринга. Функция имеет явный контракт: принимает `GoalSpec` + `OutcomeObservation`,
возвращает один `GoalSpec` (по приоритету) или `None`.

### 3. Целевой цикл не меняет `select_affordance`

Целевой цикл — отдельная функция `run_goal_cycle()`, которая вызывает
`select_goal()`, затем `select_affordance()`, затем возврат результата.
`select_affordance` (ADR-0012) не изменяется: цели не добавляются в его
сигнатуру.

### 4. Наблюдаемые исходы: `OutcomeObservation`

Строится из `Observation` (ADR-0013) через реестр предикатов:
`predicate_id → FactObservation → grade`. Не через `relevance` аффордансов.

### 5. Итерация целевого цикла

```
1. observe        → Observation (ADR-0013)
2. observe_outcome → OutcomeObservation (из Observation + реестр предикатов)
3. select_goal    → GoalSpec | None
4. select_affordance (для выбранной цели) → ScoredCandidate | None
5. (action выполняется вне Core — Shell/Host)
6. observe again  → повторный шаг 1 (после действия)
7. outcome_check  → OutcomeObservation (удовлетворён ли исход)
```

Шаги 5–6 выполняются вне Core. `run_goal_cycle` — чистая функция, вызываемая
Host.

### 6. Разделение ролей: Core / Host / Shell

| Роль | Ответственность |
|---|---|
| Core | `select_goal`, `observe_outcome`, `run_goal_cycle` (чистые функции) |
| Host | вызов `run_goal_cycle`, повторное наблюдение, лимиты, таймауты |
| Shell | выбор `GoalSpec`/`OutcomeSpec` (конфигурация), авторизация |

Core не хранит состояние между итерациями.

### 7. Реестр предикатов

`predicate_id` в `OutcomeSpec` — строка, разрешаемая через реестр (registry
pattern, ADR-0013 §3.4). Не замыкание, не словарь в модуле.

## Связанные ADR

- **ADR-0012:** окно аффордансов, скорер (`select_affordance` остаётся без
  изменений в этом разрезе).
- **ADR-0013:** локальный мир, `Observation`, предикаты.
- **ADR-0015:** форма pragmatic для целей (раздел 3).

## Что не входит

- Реализация `pragmatic` (ADR-0015 §3).
- Подключение к `host/loop.py` — отдельный шаг.
- Обучение приоритетов.
