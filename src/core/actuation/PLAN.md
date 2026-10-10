# PLAN.md — src/core/actuation

Реализация `src/core/actuation/SPEC.md` (S8, этапы 1–5).

## Файлы

- `models.py` — `OptionSource`, `Option`, `OptionWindow`, `OptionContext`,
  `ActuationPreferences`, `OptionCandidate`, `OptionTrace`, `Fact`, `Guard`,
  `Regularity`, `Effect`, `ToolAnnotations`, `ActuationKind`, `ActuationStatus`,
  `Actuation`, `ActuationResult`, `NodeKind`, `NodeStatus`, `Node`, `TickContext`,
  `TickMemory`, `Goal` (frozen, с валидацией)
- `compute.py` — `build_options`, `score_option`, `select_option`,
  `evaluate_fact`, `guard_holds`, `regularity_cost`, `classify_reversible`,
  `order_children`, `tick`, `backward_chain` (Core, чистые)
- `facts.py` — реестр `FACTS` (единый источник имён) + `DEFAULT_FACT_VALUE`
- `__init__.py` — re-export
- `src/tests/test_actuation.py` (этап 1), `src/tests/test_actuation_facts.py`
  (этап 2), `src/tests/test_actuation_effects.py` (этап 3),
  `src/tests/test_actuation_bt.py` (этап 4),
  `src/tests/test_actuation_generator.py` (этап 5)

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

### Этап 3 (следующий)

1. `Effect` (frozen): `fact`, `value ∈ [0, 1]`; валидация.
2. `ActuationKind` (Enum: SPEAK/INVOKE_TOOL), `ActuationStatus`
   (Enum: RUNNING/SUCCESS/FAILURE/PREEMPTED).
3. `Actuation` (frozen): `kind`, `goal`, `payload`; валидация непустого goal.
4. `ActuationResult` (frozen): `status`, `data`.
5. `ToolAnnotations` (frozen): консервативные дефолты (`read_only=False`,
   `destructive=True`, `idempotent=False`, `open_world=True`).
6. `classify_reversible(annotations, *, trusted)` — недоверенный → `False`;
   доверенный → `read_only and not destructive`.
7. `Option.guard: Guard | None = None`, `Option.effect: Effect | None = None` —
   аддитивно (дефолт `None`, совместимость с этапами 1–2).
8. Тесты: валидация, консервативный дефолт, доверие (trusted), аддитивность
   (опция без guard/effect работает как раньше), enum-значения.
9. Re-export в `__init__.py` и `src/core/__init__.py`.

### Этап 4 (следующий)

1. `NodeKind` (Enum: CONDITION/ACTION/SEQUENCE/FALLBACK), `NodeStatus`
   (Enum: RUNNING/SUCCESS/FAILURE).
2. `Node` (frozen): `kind`, `name`, `guard`, `actuation`, `children`,
   `regularities`; валидация формы (лист ↔ guard/actuation, композит ↔ children).
3. `TickContext` (frozen): `facts`, `action_status` (исходы действий от Shell).
4. `TickMemory` (frozen): `running_path`.
5. `order_children(children, facts)` — устойчивая сортировка по
   `regularity_cost` (тай-брейк — исходный порядок).
6. `tick(node, context, memory, path)` → `(NodeStatus, TickMemory)`:
   - CONDITION → `guard_holds`; ACTION → статус из контекста (дефолт RUNNING);
   - SEQUENCE → порядок, resume по памяти, FAILURE останавливает;
   - FALLBACK → приоритет, SUCCESS останавливает, преемпция бегущего.
7. Тесты: листья, sequence (resume/failure), fallback (приоритет/преемпция),
   `order_children`, детерминизм, валидация формы, тотальность.
8. Re-export в `__init__.py` и `src/core/__init__.py`.

### Этап 5 (следующий)

1. `Goal` (frozen): `fact`, `value ∈ [0, 1]`; валидация.
2. `backward_chain(goal, options, state, *, max_depth=3)` → `Node`:
   - цель уже истинна → `Condition`-узел на цель;
   - выбор опции по `effect.fact.name == goal.fact.name` (тай-брейк — окно);
   - опция с `guard` → `Sequence(subtree(guard), action)`;
   - нет опции / исчерпан `max_depth` → безопасный отказ (`Condition`);
   - `max_depth < 1` → `ValueError`.
3. Тесты: цепочка (2 шага), уже истинно, недостижимо, горизонт, детерминизм,
   ablation «убрать effect → схлопывается», валидация.
4. Re-export в `__init__.py` и `src/core/__init__.py`.

## Заметки

- `build_options` принимает `Affordance` из `src.mcp.probe` — **не** копирует модель.
- `score_option` — чистая функция: нет побочных эффектов, нет доступа к runtime-состоянию.
- `irreversible_penalty=0.5` — стартовая величина (SPEC §Open Questions).
- `relevance` как обогащение: если `None` → дефолт `uncertainty`; если задан — `uncertainty·relevance`.
- **Этап 2:** `state` — снимок фактов (`Mapping[str, float]`), измеряет Shell; Core чист.
- `evaluate_fact` **тотален** по имени: неизвестный факт → `fact.default` (не падение).
- Реестр `FACTS` — единый источник имён: Core и Shell не расходятся молча.
- Значения фактов ∈ [0, 1]; категориальные факты (режим) не тащим.
- Связывание фактов с опциями (`Option.guard`/`effect`) — **этап 3** (реализовано).
- **Этап 3:** `Effect` — символьная дельта (планирование), данные — `ActuationResult`.
- `classify_reversible`: Core не знает `Provenance`; Shell транслирует в `trusted`.
- `Actuation`/`ActuationResult` — общий статус, разный payload; исполнение — Shell.
- Прокидка аннотаций MCP — отдельный шаг (правка `mcp/client.py`), не здесь.
- **Этап 4:** BT — frozen-данные + свободные функции (не иерархия классов).
- `tick` реактивен: с корня каждый тик; `TickMemory` хранит один `running_path`.
- Исходы действий входят в Core через `TickContext.action_status` (инжектит Shell).
- `FALLBACK` даёт преемпцию: более приоритетный ребёнок вытесняет бегущего.
- **Этап 5:** дерево выводится `backward_chain` от цели, не пишется руками.
- Недостижимость/горизонт → безопасный отказ (`Condition`), не фиктивный `Action`.
- Опция с `guard` → подцель предваряет действие (`Sequence`); так возникает
  двухшаговость. LLM не участвует (ADR-0007).