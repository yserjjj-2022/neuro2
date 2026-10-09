# PLAN.md — src/core/actuation

Реализация `src/core/actuation/SPEC.md` (S8, этапы 1–2).

## Файлы

- `models.py` — `OptionSource`, `Option`, `OptionWindow`, `OptionContext`,
  `ActuationPreferences`, `OptionCandidate`, `OptionTrace`, `Fact`, `Guard`,
  `Regularity` (frozen, с валидацией)
- `compute.py` — `build_options`, `score_option`, `select_option`,
  `evaluate_fact`, `guard_holds`, `regularity_cost` (Core, чистые)
- `facts.py` — реестр `FACTS` (единый источник имён) + `DEFAULT_FACT_VALUE`
- `__init__.py` — re-export
- `src/tests/test_actuation.py` (этап 1), `src/tests/test_actuation_facts.py` (этап 2)

## Порядок

### Этап 1 (сделан)

1. `OptionSource` (Enum), `Option` (+валидация), `OptionWindow` (+ `find`/`ids`/`tools`).
2. `OptionContext`, `ActuationPreferences` (+валидация весов/порогов).
3. `OptionCandidate`, `OptionTrace`.
4. `build_options` — из `Affordance` → `Option(source=TOOL)`, порядок стабилен.
5. `score_option` — тотален: TOOL → pragmatic=0, epistemic=uncertainty·relevance (если relevance ≥ floor).
6. `select_option` — детерминированный argmax + тай-брейк (индекс в окне), полная трасса.
7. Тесты: открытость окна, тотальность скорера, детерминизм/тай-брейк, необратимая занижена, валидация.
8. Re-export в `src/core/__init__.py`.

### Этап 2 (следующий)

1. `Fact` (frozen): `name`, `default`, `description`; валидация `default ∈ [0, 1]`.
2. `Guard` (frozen): `fact`, `threshold ∈ [0, 1]`; валидация.
3. `Regularity` (frozen): `fact`, `weight >= 0`; валидация.
4. `facts.py`: `DEFAULT_FACT_VALUE = 0.0` + реестр `FACTS` (начальный набор).
5. `evaluate_fact(fact, state) = clip(state.get(fact.name, fact.default), 0, 1)`.
6. `guard_holds(guard, state) = evaluate_fact(guard.fact, state) >= guard.threshold`.
7. `regularity_cost(regs, state) = Σ weight·evaluate_fact(fact, state)`; пусто → 0.0.
8. Тесты: тотальность (неизвестный факт → дефолт), клип, границы порога,
   монотонность, пустой вход, детерминизм, валидация, реестр.
9. Re-export в `__init__.py` и `src/core/__init__.py`.

## Заметки

- `build_options` принимает `Affordance` из `src.mcp.probe` — **не** копирует модель.
- `score_option` — чистая функция: нет побочных эффектов, нет доступа к runtime-состоянию.
- `irreversible_penalty=0.5` — стартовая величина (SPEC §Open Questions).
- `relevance` как обогащение: если `None` → дефолт `uncertainty`; если задан — `uncertainty·relevance`.
- **Этап 2:** `state` — снимок фактов (`Mapping[str, float]`), измеряет Shell; Core чист.
- `evaluate_fact` **тотален** по имени: неизвестный факт → `fact.default` (не падение).
- Реестр `FACTS` — единый источник имён: Core и Shell не расходятся молча.
- Значения фактов ∈ [0, 1]; категориальные факты (режим) не тащим.
- Связывание фактов с опциями (`Option.guard`/`effect`) — **этап 3**, не здесь.
