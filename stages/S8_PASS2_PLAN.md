# PLAN.md — S8, проход 2: Целевой исполнительный контур в runtime

Реализация `stages/S8_PASS2_SPEC.md`. Предмет — **замыкание целевого исполнительного
контура S8 в рабочем runtime**; стенд A/B — средство проверки. Порядок: сначала
Core (чистые функции + тесты), затем Config, затем Shell (wiring в `HostLoop`),
затем сквозные тесты на собранном хосте через рабочий цикл. Модульные SPEC
(`src/core/actuation/`, `src/host/`, `src/config/`, `src/telemetry/`)
обновляются по ходу.

Порядок SDD (CONSTITUTION §5.1): SPEC → PLAN → задачи → код. Этот PLAN
следует после `stages/S8_PASS2_SPEC.md`; ADR — шаг 0.

## Файлы

### Новые

1. `src/host/goal_director.py` — `Environment` (Protocol), `GoalAttempt`,
   `GoalDirector` (Shell: наблюдение → цель → цепочка → исполнение → проверка).
2. `src/tests/test_goal_core.py` — Core: `unsatisfied_outcomes`, `select_goal`,
   `goal_reached`, `effect_gain`, pragmatic в `score_option`, ранжирование в
   `backward_chain`.
3. `src/tests/test_goal_director.py` — Shell: жизненный цикл (RUNNING, повторы,
   cooldown, тайм-аут, смена цели, отказ Gate).
4. `src/tests/test_goal_ab_stand.py` — сквозные приёмочные тесты на собранном
   хосте через `HostLoop.run`/`step_once` + стенд A/B (включая контроль F).
5. `src/tests/ab_environment.py` — тест-фикстура стенда A/B (`AbEnvironment`).

### Изменяемые

1. `src/core/actuation/models.py` — `PreferredOutcome`; `OptionContext.goal`/
   `.facts` (дефолты).
2. `src/core/actuation/compute.py` — `unsatisfied_outcomes`, `select_goal`,
   `goal_reached`, `effect_gain`; pragmatic в `score_option`; опц. ранжирование
   в `backward_chain`.
3. `src/core/actuation/__init__.py`, `src/core/__init__.py` — re-export.
4. `src/config/params.py` — `GoalConfig` + `OptionDeclaration`; поля
   `HostConfig.goal`/`HostConfig.option_declarations`; валидация.
5. `src/config/__init__.py`, `src/config/SPEC.md` — re-export/доки.
6. `src/host/loop.py` — `goal_director` в `HostLoop`; вызов в `step_once`;
   `build_host_loop` собирает `GoalDirector`; `explore()` подавляется при
   активном S8 (§8 SPEC).
7. `src/host/SPEC.md`, `src/host/PLAN.md` — новый Shell-модуль и wiring.
8. `src/telemetry/models.py` + `logger.py` — поля `selected_goal`, `goal_reason`,
   `goal_attempt`, `goal_achieved`, `goal_call_status`, `active_specialization`.
9. `src/telemetry/SPEC.md`.
10. `src/__main__.py` — флаг включения целевого контура (напр. `--goal`).
11. `src/core/actuation/SPEC.md` — новые Core-типы/функции (PreferredOutcome,
   pragmatic относительно цели).
12. `BUILD_ROADMAP.md`, `VALIDATION.md`, `SPECS.md`, `BACKLOG.md`, `README.md`
    — синк; `adr/0014-*.md`.

## Зависимости

- **Внешние:** новых нет (stdlib + `numpy`; `mcp` уже есть).
- **Внутренние:** `core/actuation` (чистый, без `host`/`config`);
  `host/goal_director` ← `core/actuation`, `host/executor`, `host/probe`, `config`;
  `config` ← `core/actuation` (как `ActuationConfig`). Циклов нет.

## Порядок работ

### Шаг 0. ADR-0014 + ADR-0015 + обновление модульных SPEC

1. ADR-0014: цель из исходов+фактов (не из колонки); `PreferredOutcome` →
   `Goal`; `select_option`/`backward_chain` как селектор (не `select_affordance`);
   два вида успеха; одно действие в полёте; сосуществование S6/S8; именование
   `active_specialization`/`selected_goal`; разделение `GoalConfig`
   (цели/приоритеты) и `OptionDeclaration` (семантика опции по устойчивому ID).
2. ADR-0015: pragmatic — доля закрытия разрыва `P`; `effect.value` —
   прогнозируемое **итоговое** значение факта (не приращение); единый скаляр
   ранжирования генератора; подцель оценивается относительно себя; неизвестный
   факт → `fact.default` (без «пустой снимок → 0»).
3. Обновить `src/core/actuation/SPEC.md` (этап 6: Core-типы/функции),
   `src/host/SPEC.md` (GoalDirector/wiring) и `src/config/SPEC.md`
   (`GoalConfig`/`OptionDeclaration`) до кода.

### Шаг 1. Core: цель и прагматика

1. `PreferredOutcome` (frozen, валидация `target ∈ [0, 1]`, `priority >= 0`).
2. `unsatisfied_outcomes` (по приоритету ↓, тай-брейк — порядок).
3. `select_goal` (первый неудовлетворённый → `Goal`; иначе `None`).
4. `goal_reached` (через `evaluate_fact`/`guard_holds`).
5. `effect_gain` — доля закрытия разрыва `P`; `effect.value` = прогнозируемое
   **итоговое** значение факта ∈ [0, 1] (не приращение); чужой эффект / нет
   эффекта / `d_до <= ε` → 0.
6. `OptionContext.goal`/`.facts` (дефолты) + pragmatic в `score_option`
   (без цели → 0, совместимость).
7. `backward_chain` — опц. `context`/`preferences`: лучшая опция по скореру
   вместо «первой по окну»; без них — прежнее поведение.
8. Тесты Core: тотальность, деградация неизвестного факта, тай-брейк,
   детерминизм, аддитивность (без цели → прежний `score_option`), форма `P`
   (в т.ч. `effect.value` = итог, `d_до <= ε`).

### Шаг 2. Config: GoalConfig + OptionDeclaration

1. `GoalConfig` (frozen): `enabled`, `outcomes`, `max_attempts`,
   `deadline_ticks`, `cooldown_ticks`; валидация.
2. `OptionDeclaration` (frozen): `option_id`, `effect`, `guard`; валидация
   (непустой `option_id`, хотя бы одно из `effect`/`guard`).
3. Поля `HostConfig.goal` (дефолт `enabled=False`) и
   `HostConfig.option_declarations`.
4. **Валидация при сборке** (`build_host_loop`): `option_id` разрешается в
   существующую опцию окна; факты согласованы с исходами; дубликаты запрещены →
   `ValueError` (fail-fast).
5. Тесты: дефолты, валидация (`max_attempts >= 1`, тики `>= 0`), несогласованный
   `option_id` → `ValueError`, совместимость (`enabled=False`).

### Шаг 3. Shell: GoalDirector + wiring

1. `Environment` (Protocol) + `GoalAttempt`.
2. `GoalDirector.tick`: снимок фактов → `select_goal` → окно (мемоизированный
   снимок) + `OptionDeclaration` (наложение `replace` по `option_id`) →
   `backward_chain` по цели → `tick_actuation` до терминала → re-observe →
   `goal_reached`.
3. Жизненный цикл (§7 SPEC): кэш дерева на попытку, `cooldown_ticks`, повторы
   `max_attempts`, `deadline_ticks` → преемпция, смена цели → новое дерево.
4. Wiring: `HostLoop.goal_director`; вызов в `step_once`; `build_host_loop`
   собирает директор при `goal.enabled` и `actuation.enabled`.
5. Сосуществование: `explore()` подавляется при наличии цели/действия в полёте.
6. Телеметрия: аддитивные поля (в т.ч. `active_specialization` — трасса).
7. Тесты Shell: RUNNING не пересобирает дерево; повтор запрещён до терминала;
   cooldown; тайм-аут → FAILURE; отказ Gate → FAILURE + причина; смена цели в
   полёте; `enabled=False` → прежний контур.

### Шаг 4. Стенд A/B + сквозные приёмочные тесты

1. `AbEnvironment` (тест-фикстура): `observe()` → `{a_serviced, b_serviced}`;
   `probe_fn` → `service_a`/`service_b`; флаги `forbid`/`lie`.
2. `test_goal_ab_stand.py` через **рабочий цикл** (`HostLoop.run`/`step_once`):
   сценарии из §13 SPEC (только A; только B; оба обслужены; запрет; успех без
   изменения; многотиковое действие; смена предпочтения).
3. **Контроль F:** при сопоставимом возбуждении A- и B-ситуации → разные
   инструменты (`service_a` vs `service_b`); образец детерминизма —
   `test_host_projection_chain.py`.

### Шаг 5. Документация и валидация

1. `VALIDATION.md` — новый раздел (замкнутый целевой контур: сценарии §13 +
   контроль F).
2. Синк `BUILD_ROADMAP.md`, `SPECS.md`, `BACKLOG.md`, `README.md`.
3. `ruff check . && ruff format .`; `pytest`; pyright strict для новых Core.

## План тестов

| Тест | Покрытие | Инвариант SPEC |
|---|---|---|
| `test_goal_core.py` | `select_goal`/`unsatisfied_outcomes`/`goal_reached`/`effect_gain`/pragmatic/ранжирование | 1–3, 8 |
| `test_goal_director.py` | жизненный цикл: RUNNING, повторы, cooldown, тайм-аут, смена цели, отказ Gate | 5, 7, 9 |
| `test_goal_ab_stand.py` | сквозные сценарии §13 + контроль F | 4, 6, 10 |
| `test_config.py` (расширение) | `GoalConfig` валидация/дефолты | 8 |
| `test_telemetry_*` (расширение) | сериализация новых полей | — |

## Заметки

- **Стенд — не предмет:** имена A/B только в `src/tests/ab_environment.py`.
- **MVP-простота:** одно действие в полёте, ограниченные повторы, без
  `Parallel`/сложной преемпции.
- **Не трогаем финализированные тесты S7/S8** — новый контур отдельный,
  включается флагом; дефолт `enabled=False` → прежний контур идентичен.
- **Детерминизм:** синтетические часы + fake-meter + детерминированный
  `probe_fn`; окно — мемоизированный снимок (ADR-0012 §9).
- **Границы:** без обучения, без вероятностной модели мира, без изменений
  сенсорной математики.

## Статус

План зафиксирован (обсуждение 2026-10-11). Код — после утверждения SPEC/PLAN.

**Реализовано** (2026-10-11): Шаги 0–4. Core (`PreferredOutcome`, `select_goal`,
`unsatisfied_outcomes`, `goal_reached`, `effect_gain`, pragmatic в `score_option`,
ранжирование `backward_chain`); Config (`GoalConfig`, `OptionDeclaration`);
Shell (`host/goal_director.py`, wiring в `HostLoop`, `explore()` подавляется);
стенд A/B + сквозные тесты §13 + контроль F; телеметрия (6 полей).

**Ревью-проход** (2026-10-11): закрыты дефект жизненного цикла и пробелы проверок:

1. **Дефект обработки завершения:** активная попытка разрешается **первой** —
   проверка достижения её цели, затем выбор следующей; «цель достигнута в полёте»
   (до терминального `poll`) фиксируется собственным событием, не молчаливой
   сменой цели (`GoalDirector.tick`/`_achieve`).
2. **Gate deny ≠ отказ сервиса:** в стенде разделены (`ab_gate_denying` отзывает
   право → `calls` пуст; `forbid` → вызов есть); два отдельных теста.
3. **Контроль F симметричен и численен:** «не обслужен только A» vs «только B»,
   сверка `F` на тике выбора; **детерминизм** — полная трасса по тикам + вызовы.
4. **Трасса запуска привязана к факту:** `GoalAttempt.chain_goal` (корень
   цепочки) и `launched_goal` (фактический лист) разделены; «launched via»
   ссылается на реальный запуск.
5. **Остановка внешней операции при смене цели:** тест на преемпцию эффектора
   (долгий вызов старой цели не выполняется).
6. `build_host_loop(gate=...)` — инъекция gate для проверки deny; публичный
   `HostLoop.last_free_energy` для численного контроля F.

Гейт: `ruff check`/`format`, `pyright` (`goal_director.py` — 0 ошибок),
`pytest` 1419 passed, 1 skipped.
