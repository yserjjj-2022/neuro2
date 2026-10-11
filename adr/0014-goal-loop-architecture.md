# ADR-0014: Целевой контур — цель из снимка фактов, `select_goal` по приоритету

## Status
Accepted

## Date
2026-10-11

## Context

Целевой цикл (goal loop) — отсутствующая часть архитектуры: `affordance`-скоринг
(ADR-0012) оценивает, что можно сделать; проекции колонок (ADR-0013) обеспечивают
различную обработку сегментов шины. Между ними нет связки, которая **формирует
цель из наблюдаемого состояния, выбирает опцию/цепочку, исполняет и проверяет
результат**.

Существующий S6-fallback (`select_affordance` в `host/loop.py`) не меняется
в этом разрезе — целевой контур подключается отдельным маршрутом.

## Decision

### 1. Простая модель цели: `PreferredOutcome` + `Goal`

Цель формируется из **предпочтительных исходов** и **снимка фактов**
`Mapping[str, float]` (не из тега победившей колонки, не из `relevance`
аффордансов).

```python
@dataclass(frozen=True)
class PreferredOutcome:
    fact: Fact
    target: float = 1.0     # желаемое значение факта ∈ [0, 1]
    priority: float = 1.0   # порядок при нескольких неудовлетворённых, >= 0

def select_goal(
    outcomes: Sequence[PreferredOutcome], facts: Mapping[str, float]
) -> Goal | None:
    """Первый неудовлетворённый по приоритету ↓ → Goal(fact, target);
    None — все исходы достигнуты или outcomes пуст.
    Тай-брейк — порядок объявления в outcomes."""
```

`Goal` — простой тип `Fact` + целевое значение (`value ∈ [0, 1]`). Не вводим
дополнительные слои `GoalSpec`/`OutcomeSpec`/`predicate_id` — они могут быть
полезны позже, но сейчас добавляют ненужный перевод между фактом и целью.

### 2. `selected_goal` — результат `select_goal()`, не побочный эффект скоринга

`select_goal()` принимает `PreferredOutcome` + снимок фактов, возвращает один
`Goal` (по приоритету) или `None`. Диапазон `priority >= 0`; тай-брейк при равных
приоритетах — **порядок объявления** в `outcomes`. Обе величины зафиксированы так
же, как в `stages/S8_PASS2_SPEC.md` §4.

### 3. Выбор действия: `select_option` / `backward_chain` → исполнитель

Целевой контур **не использует** `select_affordance` (S6) как селектор.
Селектор целевого контура — `select_option()` + `backward_chain()` (ADR-0012
§5–§6) → `tick_actuation` → эффекторы.

S6-маршрут (`select_affordance` в `host/loop.py`) остаётся отдельным,
неизменённым. Целевой контур подключается параллельно.

### 4. Снимок фактов — `Mapping[str, float]`, не `Observation` + реестр предикатов

Целевой контур работает с **снимком фактов** `Mapping[str, float]`, который
Shell собирает через `Environment.observe()`. Core-функция `evaluate_fact(fact,
state)` извлекает значение по имени факта с дефолтом (`fact.default`), не
требует реестра предикатов.

Ссылка на ADR-0013: это решение о **проекции колонок через именованные reads
сегментов** (различная обработка сегментов шины), а не о «локальном мире».
Снимок фактов для целевого контура — общий механизм Shell, не часть ADR-0013.

### 5. Итерация целевого цикла

```
1. observe        → Mapping[str, float] (снимок фактов)
2. unsatisfied_outcomes → tuple[PreferredOutcome, ...]
3. select_goal    → Goal | None
4. build_options + OptionDeclaration → OptionWindow + декларации
5. score_option + select_option + backward_chain(goal, options, facts, context) → Node
6. tick_actuation (Shell: Gate → Effector)
7. re-observe → снимок фактов → goal_reached(goal, new_facts)
```

Шаги 1, 6–7 — Shell. Шаги 2–5 — Core (чистые функции).

### 6. Разделение ролей: Core / Host / Shell

| Роль | Ответственность |
|---|---|
| Core | `unsatisfied_outcomes`, `select_goal`, `goal_reached`, `effect_gain`,
  `score_option`, `select_option`, `backward_chain` (чистые функции) |
| Host | `GoalDirector.tick()`, повторное наблюдение, лимиты, таймауты,
  преемпция |
| Shell | `Environment.observe()`, `build_options`, `CapabilityGate`,
  `ActuatorExecutor`, `GoalConfig`/`OptionDeclaration` (конфигурация) |

Core не хранит состояние между итерациями.

### 7. Валидация деклараций при сборке

`HostConfig.option_declarations` — `tuple[OptionDeclaration, ...]` по устойчивому
`option_id` (`tool:<name>`). `build_host_loop` валидирует: каждый `option_id`
разрешается в существующую опцию окна; `effect.fact`/`guard.fact` присутствуют в
объявленной схеме состояния (не каждый факт обязан быть конечной целью —
вспомогательные и промежуточные допустимы для многошаговых цепочек); дубликаты
запрещены. Несогласованность → `ValueError` (fail-fast).

## Связанные ADR

- **ADR-0012:** окно аффордансов, скорер, BT, генератор (`select_option`/
  `backward_chain` — целевой селектор).
- **ADR-0013:** проекции колонок через именованные reads сегментов (ссылка
  исправлена: это не «локальный мир/предикаты»).
- **ADR-0015:** форма pragmatic для целей (доля закрытия разрыва).

## Что не входит

- `GoalSpec`/`OutcomeSpec`/реестр предикатов — могут быть введены позже
  отдельно.
- Подключение к `host/loop.py` — отдельный шаг.
- Обучение приоритетов.
