# PLAN.md — S6: Автономия

Реализация `stages/S6_SPEC.md`. Порядок — два прохода: **проход 1** даёт
Core-механики автономии (метакогниция, critical slowing down, классификатор,
сброс, консолидация, драйв, факторизация), **проход 2** — длинный горизонт
(C10), ночной цикл, MCP-зондирование. **HITL-сброс и самоотчёт вынесены в
отдельный этап S7** (ADR-0010; `stages/S7_PLAN.md`). Каждый шаг — сначала Core
(чистые функции + тесты), затем Shell (wiring).

Модульные SPEC (`src/core/selfcontrol/`, `src/memory/`, `src/core/policy/`,
`src/core/factorization/`, `src/config/`, `src/telemetry/`) обновляются по
ходу. Новые модули требуют `SPEC.md` + `README.md` до кода (CONSTITUTION §5.1).

## Файлы

### Новые (проход 1)

1. `src/core/selfcontrol/` — `models.py` (`Metacognition`,
   `CriticalSlowingDown`, `ChangeKind`, `ChangeAssessment`, `ResetLevel`,
   `ResetPlan`), `compute.py` (`compute_conflict`, `compute_metastability`,
   `compute_saturation`, `critical_slowing_down`, `classify_change`,
   `plan_reset`), `monitor.py` (`SelfMonitor`), `__init__.py`, `SPEC.md`,
   `README.md`
2. `src/core/factorization/` — `models.py` (`Factor`, `FactorizedState`),
   `compute.py` (`posterior`, `marginal`, `update_factor`, `argmax_state`),
   `__init__.py`, `SPEC.md`, `README.md`
3. `src/memory/consolidation.py` — `ConsolidationPlan`, `Schema`,
   `plan_consolidation`, `consolidate`, `ConsolidationResult`
4. `src/tests/test_selfcontrol_compute.py`
5. `src/tests/test_selfcontrol_monitor.py`
6. `src/tests/test_factorization.py`
7. `src/tests/test_memory_consolidation.py`
8. `src/tests/test_policy_explore.py`
9. `src/tests/test_autonomy_integration.py`

### Изменяемые

1. `src/core/policy/models.py` — `Action.EXPLORE`; `PolicyContext.metacognition`;
   `MetacognitionView` Protocol; `Preferences.explore_threshold`
2. `src/core/policy/compute.py` — `_evaluate_explore`; `_ACTION_ORDER`
3. `src/config/params.py` — `AutonomyConfig`; поле в `HostConfig`; валидация
4. `src/config/__init__.py`, `src/config/SPEC.md` — re-export/доки
5. `src/memory/store.py` — `delete(ids)`; таблица `schemas` (+ `save_schema`)
6. `src/memory/SPEC.md` — консолидация
7. `src/telemetry/models.py` + `logger.py` — `metacog_conflict`,
   `metacog_metastability`, `metacog_saturation`, `reset_level`, `change_kind`,
   `consolidated_pruned`
8. `src/telemetry/SPEC.md`
9. `src/host/loop.py` — selfcontrol wiring, `record_metacognition`, телеметрия
10. `src/host/SPEC.md`
11. `src/__main__.py` — `--no-autonomy`, `--consolidate`
12. `src/tests/test_config.py`, `test_telemetry_*`, `test_policy_compute.py` —
    адаптация + новые случаи
13. `BUILD_ROADMAP.md`, `VALIDATION.md`, `SPECS.md`, `BACKLOG.md`, `README.md`
    — синк

## Зависимости

- **Внешние:** только stdlib + `numpy` (уже есть). **Новой зависимости нет**
  (`pymdp` не вводится — ADR-0009 §6).
- **Внутренние:** `selfcontrol` ← `numpy`, `attractors` (`Vector`); `policy` ←
  `selfcontrol` **только через Protocol** (без цикла); `consolidation` ←
  `memory` (Episode/cosine); `factorization` — автономный; `host.loop` ←
  `selfcontrol`, `consolidation`, `factorization`.

## Проход 1 — Core-механики автономии (P0)

### Шаг 1. selfcontrol Core: наблюдаемые

1. `Metacognition` (frozen): `conflict`, `metastability`, `epistemic_uncertainty`,
   `saturation` ∈ [0, 1] (валидация).
2. `compute_conflict(scores)` — нормированный разброс (энтропия/CV) ∈ [0, 1].
3. `compute_metastability(switch_flags, *, window)` — частота смен ∈ [0, 1].
4. `compute_saturation(f_trend)` — доля/тренд ∈ [0, 1].
5. Тесты: границы, пустой вход, детерминизм, монотонность.

### Шаг 2. selfcontrol Core: critical slowing down

1. `CriticalSlowingDown` (frozen): `variance`, `autocorrelation`, `slowing`,
   `is_warning`.
2. `critical_slowing_down(series, *, variance_gain, autocorr_gain)` — z-нормировка
   дисперсии и lag-1 автокорреляции → `sigmoid` → [0, 1].
3. Тесты: короткий ряд → безопасный дефолт; рост дисперсии/автокорреляции →
   рост slowing; константный ряд → низкий slowing; детерминизм.

### Шаг 3. selfcontrol Core: классификатор + сброс

1. `ChangeKind` (STABLE/DEVELOPMENT/DRIFT), `ChangeAssessment`.
2. `classify_change(*, core_preserved, traceable, coherent, changed=True)` —
   три оси INTENT §4.
3. `ResetLevel` (SOFT/FREEZE/HARD), `ResetPlan`.
4. `plan_reset(*, slowing, assessment, soft_threshold, hard_core_broken)` —
   core нарушен → HARD; DRIFT+warning → FREEZE; warning → SOFT.
5. Тесты: три оси → DEVELOPMENT; любая нарушена → DRIFT; STABLE при
   `changed=False`; три уровня сброса; core не сбрасывается (HARD ≠ reset).

### Шаг 4. selfcontrol Shell: SelfMonitor

1. `SelfMonitor(*, window, csd_gains, warning_threshold, soft_threshold)` —
   кольцевые буферы F и switch-флагов.
2. `observe(*, scores, switched, f, partner_uncertainty) ->
   (Metacognition, ResetPlan)` — считает наблюдаемые + CSD + классификацию +
   план сброса; Shell хранит `last_change`.
3. Классификация: `core_preserved`/`traceable`/`coherent` — из входов Shell
   (детектор дрейфа, телеметрия); по умолчанию консервативно (не DRIFT).
4. Тесты: наблюдаемые копятся; warning при росте дисперсии; сброс не
   триггерится на ровном ряду.

### Шаг 5. Память: консолидация

1. `MemoryStore.delete(ids) -> int` — удаление эпизодов (+ векторов) в
   транзакции; fail-fast при закрытом store.
2. `MemoryStore.save_schema(centroid, member_count, summary) -> int` + таблица
   `schemas`.
3. `plan_consolidation(episodes, *, min_weight, schema_threshold, max_schemas,
   now) -> ConsolidationPlan` (Core, чистая): pruning + жадная кластеризация.
4. `consolidate(store, *, ...) -> ConsolidationResult` (Shell): исполнить план,
   логировать число удалённых.
5. Тесты: план детерминирован; pruning удаляет только низкий вес; схемы
   строятся из близких; Shell удаляет и логирует; recall после консолидации.

### Шаг 6. Policy: эпистемический драйв

1. `Action.EXPLORE`; `_ACTION_ORDER` дополнен.
2. `Preferences.explore_threshold` ∈ [0, 1]; валидация.
3. `_evaluate_explore(context, preferences)` — эпистемическая ценность при
   отсутствии сообщения и `metacognition.epistemic_uncertainty >= threshold`.
4. `_evaluate_identify` — частный случай (агентная неопределённость).
5. `PolicyContext.metacognition: MetacognitionView | None = None`.
6. Тесты: высокая неопределённость → EXPLORE выигрывает; `metacognition=None`
   → S5-совместимость; детерминизм; трасса объяснима.

### Шаг 7. Дискретный факторизованный слой

1. `Factor` (frozen): `name`, `states`, `prior`, `likelihood` (нормированы,
   валидация shape).
2. `FactorizedState`: `factors` (уникальные имена, валидация).
3. `posterior(factor)` — prior ∘ likelihood → нормировка (чистая).
4. `marginal(state, name)` — распределение фактора (чистая).
5. `update_factor(state, name, observation)` — байесовское обновление одного
   фактора, остальные не трогаются (чистая).
6. `argmax_state(factor)` — MAP-состояние (чистая).
7. Тесты: posterior нормирован; маргинал; обновление изолировано; MAP;
   валидация; детерминизм.

### Шаг 8. Конфиг + телеметрия

1. `AutonomyConfig` (frozen) + валидация; поле `autonomy` в `HostConfig`.
2. Телеметрия: +6 полей (метакогниция/сброс/классификация/консолидация),
   keyword-only с дефолтами.
3. `HostLoop`: `selfcontrol` (опционально), `record_metacognition`; поля в
   `logger.log`.
4. Тесты: `AutonomyConfig` валидация; телеметрия сериализуется; дефолты;
   `autonomy.enabled=False` → S5-совместимость.

### Шаг 9. Валидация прохода 1

1. Интеграционный тест: наблюдаемые копятся на тиках; консолидация через Shell
   логируется; `autonomy.enabled=False` → контур идентичен.
2. Синк `VALIDATION.md` §4 (S6 проход 1), `BUILD_ROADMAP.md` §1,
   `SPECS.md`, `BACKLOG.md`, `README.md`.

## Проход 2 — длинный горизонт, ночной цикл, MCP (P1)

HITL-сброс и самоотчёт из прохода 2 перенесены в S7 (ADR-0010); ниже — что
осталось.

### Шаг 10. Ночной цикл по расписанию ✅ (2026-10-04)

1. Триггер консолидации: по объёму эпизодов / простою (synthetic-время),
   а не только явный `--consolidate`.
2. Расписание в `AutonomyConfig` (`consolidate_every_ticks`,
   `consolidate_min_episodes`); чистая `should_consolidate(...)` в
   `memory/consolidation.py`; Shell `_maybe_consolidate` в конце тика,
   логирование в том же тике (инвариант 6).
3. Тесты: `TestConsolidationTrigger` (5), `TestNightCycle` (2).

### Шаг 11. Длинный горизонт C10 ✅ (2026-10-04)

1. Прогон C10: 1500 тиков, synthetic-часы, `seed=0`, fake-эмбеддер,
   автономия+память.
2. Критерий: нет тихого дрейфа (поведение связно), отпечаток сравнивается с
   эталоном (VALIDATION §4, S6).
3. Канал метакогниции в шину: **решение — не вводить.** Прогон показал, что
   `conflict` быстро выходит в насыщение (1.0), а прямой канал создал бы
   circularity без пользы; метакогниция остаётся полем `PolicyContext`.
4. Тесты: `test_behavioral_regress.py::TestLongHorizon` — нет дрейфа,
   ограниченность, активность не коллапсирует, детерминизм.

### Шаг 12. MCP-зондирование ✅ (2026-10-04)

1. Карта аффордансов `mcp/probe.py` (`Affordance`/`AffordanceMap`) +
   чистая `select_affordance` по неопределённости; `Action.EXPLORE`
   получает реальный эффектор `host/probe.py::ProbeEffector`.
2. Гардрейлы: capability gate (обратимое T3, необратимое T4/HITL),
   fail-safe deny; сбой транспорта не роняет тик.
3. Тесты: `test_mcp_probe.py` (7), `test_host_probe.py` (9) — зондирование
   в пределах прав; отказ вне прав/без HITL; телеметрия.

## План тестов

| Тест | Покрытие | Инвариант |
|---|---|---|
| `test_compute_conflict_*` | наблюдаемые Core | границы [0, 1] |
| `test_compute_metastability_*` | наблюдаемые Core | частота смен |
| `test_compute_saturation_*` | наблюдаемые Core | тренд F |
| `test_critical_slowing_down_*` | CSD Core | рост дисперсии/AC → warning |
| `test_classify_change_*` | классификатор | три оси INTENT §4 |
| `test_plan_reset_*` | сброс | три уровня, core не сбрасывается |
| `test_self_monitor_observe` | selfcontrol Shell | наблюдаемые копятся |
| `test_plan_consolidation_*` | консолидация Core | pruning + схемы |
| `test_store_delete` | память Shell | явное удаление |
| `test_consolidate_*` | консолидация Shell | логирование, инвариант 6 |
| `test_policy_explore_*` | драйв | EXPLORE при неопределённости |
| `test_policy_metacog_none_compat` | совместимость | S5 идентичен |
| `test_factor_posterior_*` | факторизация Core | нормировка |
| `test_factor_update_isolated` | факторизация Core | независимость факторов |
| `test_autonomy_config_*` | конфиг | валидация |
| `test_telemetry_autonomy_*` | телеметрия | сериализация |
| `test_autonomy_disabled_compat` | совместимость | S5 идентичен |

## Заметки

- **Порядок:** Core (чистые) → Shell (SelfMonitor/consolidate) → wiring.
- **Без цикла импортов:** `core.policy` ← `MetacognitionView` Protocol;
  `selfcontrol` не импортирует `policy`.
- **Расширяемость:** `PolicyContext.metacognition` — дефолт `None` (как
  `partner` в S5).
- **Инвариант 6:** удаление только через явную консолидацию + лог.
- **Обратная совместимость:** `autonomy.enabled=False` → контур S5 идентичен;
  `--no-autonomy` в CLI.
- **Детерминизм:** все Core-функции — чистые; при фиксированных входах
  одинаковые наблюдаемые/планы (replay).
- **Reference:** `Homeostat` (Core+Shell), `DriftDetector` (Shell),
  `tm` (Shell с DI), `MemoryRouter` (Shell с safe-default).
