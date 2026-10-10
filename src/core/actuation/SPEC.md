# SPEC.md — src/core/actuation

## Назначение

Ядро **секвенирования актуаций** (стадия S8, ADR-0012). Модуль отвечает на
вопрос «**какие возможности есть и какая из них ценнее сейчас**» — открытое окно
опций и тотальный скорер. Это фундамент спринта: BT/async/генератор —
надстройка над правильным выбором (`stages/ACTUATION_PLAN.md`).

Спека покрывает **этап 1** (открытое окно опций, оценка и выбор — заменяющие
нынешнее вырожденное правило `select_affordance` «первый обратимый»,
`src/mcp/probe.py`), **этап 2** (факты и кондишены «дефолт + обогащение»),
**этап 3** (эффекты, единый контракт актуации, классификация необратимости) и
**этап 4** (Behavior Tree: узлы, tick, преемпция) и **этап 5** (runtime-генератор:
`Goal`, `backward_chain` от цели, горизонт). Последующие этапы **расширят эту
спеку** перед своим кодом — здесь они не описываются (см. «Границы»).

Ключевое решение (ADR-0012): опции **не** перечисляются закрытым enum; окно
порождается **в runtime** из доступных возможностей (встроенные + MCP-тулы).
Скорер определён для **любой** опции без кода под неё. Видимость ≠ авторизация:
необратимое/неизвестное **видно** и оценено, но исполняется только через
`CapabilityGate`.

См. также:
- `adr/0012-actuation-sequencing.md` — решение (окно, скорер, дефолт+обогащение)
- `stages/ACTUATION_PLAN.md` — план S8, порядок работ
- `src/core/policy/SPEC.md` — соседний выбор (речь); **не** сливается с окном
  (вариант B: окно рядом, `Action` остаётся)
- `src/mcp/SPEC.md` — источник `Affordance` (runtime-вид тулов)

## Развилка (решено)

**Вариант B:** окно живёт **рядом** с `core/policy`, не поглощает его.
`Action`/`PolicyTrace` не трогаются (S4/S7-совместимость). На этапе 1 в окно
попадают **только тулы** (встроенные действия — отдельным шагом). Так первый срез
вообще не касается policy.

## Публичный интерфейс

```python
class OptionSource(Enum):
    TOOL = "tool"        # MCP-тул (эпистемическое зондирование)
    BUILTIN = "builtin"  # встроенное действие (этап 2+, сейчас не наполняется)


@dataclass(frozen=True)
class Option:
    id: str                       # стабильный: "tool:get_weather"
    source: OptionSource
    description: str = ""         # для привязки темы (этап 2) и трассы
    reversible: bool = False      # консервативный дефолт (ADR-0012 §4)
    cost: float = 0.0             # стоимость/латентность (оценка, не замер)
    relevance: float | None = None  # привязка темы [0,1]; None → дефолт


@dataclass(frozen=True)
class OptionWindow:
    options: tuple[Option, ...] = ()   # порядок стабилен (детерминизм)

    def find(self, option_id: str) -> Option | None: ...
    @property
    def ids(self) -> tuple[str, ...]: ...
    @property
    def tools(self) -> tuple[Option, ...]: ...


@dataclass(frozen=True)
class OptionContext:
    uncertainty: float            # [0,1] метакогнитивная неопределённость
    task: str = "none"
    mode: str = "free"


@dataclass(frozen=True)
class ActuationPreferences:
    pragmatic_weight: float = 1.0
    epistemic_weight: float = 0.5
    cost_weight: float = 0.1
    irreversible_penalty: float = 0.5   # занижение без consent (калибруется)
    relevance_floor: float = 0.0        # ниже — привязка не учитывается


@dataclass(frozen=True)
class OptionCandidate:
    option: Option
    pragmatic: float
    epistemic: float
    value: float
    reason: str


@dataclass(frozen=True)
class OptionTrace:
    chosen: Option | None         # None → окно пусто
    reason: str
    candidates: tuple[OptionCandidate, ...]  # ВСЕ рассмотренные (в т.ч. отклонённые)


def build_options(
    tools: Sequence[Affordance] = (),
    *,
    descriptions: Mapping[str, str] | None = None,
    builtin: Sequence[Option] = (),
) -> OptionWindow: ...

def score_option(
    option: Option, context: OptionContext, preferences: ActuationPreferences
) -> tuple[float, float, str]: ...

def select_option(
    window: OptionWindow,
    context: OptionContext,
    preferences: ActuationPreferences = ActuationPreferences(),
) -> OptionTrace: ...
```

### Правила оценки (этап 1)

Оценка **тотальна**: определена для любой опции, включая неизвестную.

| Источник | pragmatic | epistemic | reason |
|---|---|---|---|
| `TOOL` | `0.0` (эффект не объявлен) | `uncertainty` (дефолт); при `relevance ≥ floor` — `uncertainty·relevance` | «эпистемическая опция» / «привязка темы» |
| `BUILTIN` | (этап 2) | (этап 2) | — |

`value = pragmatic_weight·pragmatic + epistemic_weight·epistemic
− cost_weight·cost − (0 если reversible иначе irreversible_penalty)`.

`select_option` — детерминированный argmax; **тай-брейк** — порядок опций в окне
(стабильный). `reason` выводится из оценки победителя, не постфактум. В
`OptionTrace.candidates` — **все** опции (explainability, ADR-0012 §2).

### Открытость окна

`build_options` строит окно из runtime-возможностей: каждый `Affordance` →
`Option(source=TOOL, id=f"tool:{name}")`. Порядок — порядок входа (стабильный).
`descriptions` (если задан) наполняет `description` для привязки темы; иначе
пусто (дефолт сохраняет работоспособность).

## Этап 2. Факты и кондишены (дефолт + обогащение)

**Факт** — именованная градуированная закономерность мира со значением в
[0, 1] (0/1 = булево). Словарь фактов **открыт**: новый факт вводится без
правки типов. **Кондишены** — функция над парой (факт × состояние),
определённая для **любого** факта, включая неизвестный (деградация к дефолту,
не падение). Различение (ADR-0012): **жёсткий** кондишен → `Guard` (гейт,
закономерность мира, опровержима); **мягкий** → `Regularity` (стоимость,
видовая склонность).

```python
DEFAULT_FACT_VALUE = 0.0  # нейтральное значение неизвестного факта


@dataclass(frozen=True)
class Fact:
    name: str                    # стабильное имя: "network_available"
    default: float = DEFAULT_FACT_VALUE  # значение, если факт не измерен
    description: str = ""        # для трассы/аудита


@dataclass(frozen=True)
class Guard:
    fact: Fact
    threshold: float             # жёсткое: fact >= threshold → узел проходит


@dataclass(frozen=True)
class Regularity:
    fact: Fact
    weight: float = 1.0          # мягкое: вклад факта в стоимость, >= 0


# Реестр имён фактов — единый источник (Core владеет именами, Shell измеряет).
FACTS: Mapping[str, Fact] = {...}   # "network_available", "topic_bound", ...


def evaluate_fact(fact: Fact, state: Mapping[str, float]) -> float: ...
def guard_holds(guard: Guard, state: Mapping[str, float]) -> bool: ...
def regularity_cost(
    regularities: Sequence[Regularity], state: Mapping[str, float]
) -> float: ...
```

### Правила (этап 2)

- `evaluate_fact(fact, state) = state.get(fact.name, fact.default)`, клип в
  [0, 1] (значение из state вне диапазона → зажимается). **Тотальна** по имени.
- `guard_holds(guard, state) = evaluate_fact(guard.fact, state) >= guard.threshold`.
- `regularity_cost = Σ weightᵢ · evaluate_fact(factᵢ, state)`; при пустом списке
  → `0.0` (дефолт без обогащения). Монотонна по фактам (веса ≥ 0).
- **Реестр `FACTS`** фиксирует множество имён в одном месте: Core и Shell не
  расходятся молча (иначе оценка всегда по дефолту — тихий баг). Значение
  факта живёт в `state` (снимок мира от Shell), смысл/дефолт — в `Fact`.

### Открытость и границы

Словарь фактов открыт: `FACTS` — реестр известных, но `evaluate_fact` работает
с любым `Fact`. **Связывание фактов с опциями** (`Option.guard`/`Option.effect`)
— **этап 3**, здесь не вводится. `state` — снимок фактов (Shell измеряет шину/
gate/партнёра и отдаёт `Mapping[str, float]`); Core не тянет `host` и остаётся
чистым. Категориальные факты (режим) в первую версию **не тащим** — только
градуированные ∈ [0, 1].

## Этап 3. Эффекты, контракт актуации, необратимость

**Эффект — символьная дельта факта** (не данные): отвечает на вопрос генератора
(этап 5) «достигает ли действие цели?». Данные-результат — это `ActuationResult`
и маршрутизация в шину (Shell, этап 6); смешивать их с эффектом не нужно.

**Обогащение опции аддитивно:** `Option.guard`/`Option.effect` — необязательные
поля (дефолт `None`). Без них опция остаётся эпистемической (этап 1). Наличие
`effect` даёт прагматическую ценность (действие, как известно, достигает факта).

```python
@dataclass(frozen=True)
class Effect:
    fact: Fact
    value: float = 1.0          # целевое значение факта ∈ [0, 1]


class ActuationKind(Enum):
    SPEAK = "speak"
    INVOKE_TOOL = "invoke_tool"


class ActuationStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    PREEMPTED = "preempted"


@dataclass(frozen=True)
class Actuation:
    kind: ActuationKind
    goal: str                   # идентификатор цели/опции ("tool:get_weather")
    payload: str = ""           # текст речи ИЛИ имя тула; аргументы — Shell


@dataclass(frozen=True)
class ActuationResult:
    status: ActuationStatus
    data: tuple[float, ...] = ()  # данные-результат → в шину (Shell, этап 6)


@dataclass(frozen=True)
class ToolAnnotations:
    read_only_hint: bool = False
    destructive_hint: bool = True    # консервативный дефолт (ADR-0012 §4)
    idempotent_hint: bool = False
    open_world_hint: bool = True


def classify_reversible(annotations: ToolAnnotations, *, trusted: bool) -> bool: ...
```

### Правила (этап 3)

- **`Effect`** — символьная дельта (`fact := value`), `value ∈ [0, 1]`. Только для
  планирования; данные — не эффект.
- **Обогащение:** `Option.guard: Guard | None`, `Option.effect: Effect | None`.
  Аддитивно: без них опция оценивается как эпистемическая (совместимость).
- **`classify_reversible`** — чистая классификация необратимости из аннотаций и
  доверия источника:
  - **недоверенный источник** (`trusted=False`) → всегда `False` (hint не
    принимается как основание для автономии; аннотации — только подсказки,
    ADR-0012 §4);
  - **доверенный** (`trusted=True`) → `read_only_hint and not destructive_hint`.
  - Core **не** знает `Provenance`: Shell транслирует `official → trusted=True`,
    `community/local/нет → trusted=False`. Так `actuation` не импортирует
    `integrations` (без цикла зависимостей, PLAN §Зависимости).
- **`Actuation`/`ActuationResult`** — единый контракт «текст vs тул»: общий
  статус, разный payload. Исполнение (эффекторы, gate, шина) — Shell (этап 6).

### Границы этапа 3

Только Core-контракт и чистая классификация. **Прокидка аннотаций** из
`tools/list` → `ToolInfo` → `Affordance` → `Option` — **отдельный шаг** (правка
`src/mcp/client.py`, риск транспорта). Здесь `ToolAnnotations` вводится как
Core-тип, а `Affordance`/`select_affordance` не трогаются.

## Этап 4. Behavior Tree (узлы, tick)

**BT реактивен by construction** (ADR-0012 §5): tick с корня каждый тик; никакого
«второго executive». Минимальный словарь узлов: `Sequence`, `Fallback`, листья
`Condition`/`Action`. Статусы `Running`/`Success`/`Failure`. `Parallel` и
декораторы — **не** вводим до реального кейса.

**Core остаётся чистым:** дерево — frozen-данные, tick — свободная функция,
память Running передаётся явно, исходы действий инжектит Shell через контекст.
Отступление от «`Node` — Protocol» (PLAN §Шаг 4): вместо иерархии классов —
единый frozen `Node` с `NodeKind` (data-first, как весь модуль).

```python
class NodeKind(Enum):
    CONDITION = "condition"   # лист: проверка guard
    ACTION = "action"         # лист: актуация (статус — из контекста)
    SEQUENCE = "sequence"     # все дети по порядку
    FALLBACK = "fallback"     # первый успешный (priority, преемпция)


class NodeStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class Node:
    kind: NodeKind
    name: str = ""
    guard: Guard | None = None          # для CONDITION
    actuation: Actuation | None = None  # для ACTION
    children: tuple[Node, ...] = ()     # для SEQUENCE/FALLBACK
    regularities: tuple[Regularity, ...] = ()  # мягкий порядок детей


@dataclass(frozen=True)
class TickContext:
    facts: Mapping[str, float] = field(default_factory=dict)
    action_status: Mapping[str, NodeStatus] = field(default_factory=dict)


@dataclass(frozen=True)
class TickMemory:
    running_path: tuple[int, ...] = ()  # путь до бегущего листа (выход tick)


def tick(
    node: Node, context: TickContext
) -> tuple[NodeStatus, TickMemory]: ...

def order_children(
    children: Sequence[Node], facts: Mapping[str, float]
) -> tuple[Node, ...]: ...
```

### Правила (этап 4)

- **Лист `CONDITION`:** `guard_holds(guard, facts)` → `SUCCESS`/`FAILURE`.
- **Лист `ACTION`:** статус из `context.action_status[actuation.goal]`; отсутствует
  → `RUNNING` (Shell ещё не отчитался). Так исходы асинхронных действий входят в
  чистый Core (ADR-0012 §7–8).
- **`SEQUENCE`:** дети в объявленном порядке; `RUNNING` ребёнка → `RUNNING`;
  `FAILURE` → `FAILURE`; все `SUCCESS` → `SUCCESS`.
- **`FALLBACK`:** дети в порядке приоритета; `SUCCESS` ребёнка → `SUCCESS`; все
  `FAILURE` → `FAILURE`. **Преемпция:** каждый тик дерево перерешается с корня, и
  более приоритетный ребёнок, ставший `SUCCESS`/`RUNNING`, вытесняет бегущего.
- **Реактивность:** `tick` не хранит управляющего состояния между тиками —
  решение полностью определяется текущим `TickContext`. «Resume» достигается
  через `action_status`: завершённое действие Shell отчитывает как `SUCCESS`, и
  последовательность продолжается на следующем тике.
- **`TickMemory`** — **выход** `tick`: путь до бегущего листа (для Shell и
  телеметрии), пустой, если ничего не бежит.
- **`order_children`:** устойчивая сортировка по `regularity_cost`
  (мягкие предпочтения ребёнка); тай-брейк — исходный порядок (детерминизм).
  Применяется в `FALLBACK` (приоритет); `SEQUENCE` сохраняет объявленный порядок.
- **Валидация `Node`:** лист без детей и с обязательным `guard`/`actuation`;
  композит с детьми и без `guard`/`actuation`.

## Этап 5. Runtime-генератор (backward chaining)

**Дерево не пишем руками — оно выводится** (ADR-0012 §6). Decision выбирает
**цель-исход**; генератор разворачивает её в дерево backward chaining по
`effect`/`guard` опций, с горизонтом 2–3. Поведение **выводится**, не
программируется (`INTENT.md` §2). LLM не участвует (ADR-0007); аргументы тула —
забота Shell.

```python
@dataclass(frozen=True)
class Goal:
    fact: Fact
    value: float = 1.0          # целевое значение факта ∈ [0, 1]


def backward_chain(
    goal: Goal,
    options: Sequence[Option],
    state: Mapping[str, float],
    *,
    max_depth: int = 3,
) -> Node: ...
```

### Правила (этап 5)

- **Цель уже достигнута** (`guard_holds(Guard(goal.fact, goal.value), state)`) →
  `Condition`-узел на цель (успех без действия).
- **Выбор опции:** первая по порядку окна опция с `effect.fact.name ==
  goal.fact.name` (детерминированный тай-брейк — порядок окна).
- **Опция с `guard`:** `guard` — предпосылка. Строим подцель `Goal(guard.fact,
  guard.threshold)`, рекурсивно разрешаем её и **предваряем** действие:
  `Sequence(subtree, action)`. Так выводится двухшаговость.
- **Опция без `guard`:** сразу `Action`-узел.
- **Безопасный отказ (не выдумываем действие):**
  - нет опции, достигающей факта цели → `Condition`-узел на цель (провалится,
    если цель не истинна);
  - исчерпан `max_depth` (горизонт) → `Condition`-узел на цель.
  Оба случая — честное «не знаю как» без фиктивной модели мира.
- **`Action`-узел:** `Actuation(kind, goal=option.id, payload=option.id)`, где
  `kind`: `TOOL → INVOKE_TOOL`, иначе `SPEAK`.
- **Чистота и детерминизм:** `backward_chain` — чистая функция; одинаковый вход
  → одинаковое дерево. `max_depth < 1` → `ValueError`.
- **Защита от цикла:** рекурсия ограничена `max_depth` (guard, ведущий к тому же
  факту, упирается в горизонт).

## Инварианты

1. **FC/IS:** `build_options`/`score_option`/`select_option` — чистые;
   одинаковый вход → одинаковый `OptionTrace`.
2. **Открытость:** новый тул виден в окне **без кода** под него; окно не
   фильтрует опции по типу/классу.
3. **Тотальность:** `score_option` определён для любой `Option`; неизвестная
   деградирует к дефолту (эпистемическая опция), а не падает.
4. **Видимость ≠ авторизация:** все опции в `candidates`; необратимая —
   присутствует и занижена, исполнение — за `CapabilityGate` (не здесь).
5. **Детерминизм:** порядок окна стабилен, тай-брейк — индекс в окне.
6. **Валидация:** `cost < 0`, веса/пороги вне допустимого, `uncertainty` вне
   [0,1] → `ValueError`.
7. **Расширяемость:** обогащение (`relevance`, `description`) **аддитивно**;
   без него оценка корректна (дефолт).
8. **Тотальность кондишенов (этап 2):** `evaluate_fact`/`guard_holds`/
   `regularity_cost` определены для любого факта; неизвестный деградирует к
   `fact.default`, не падает.
9. **Монотонность (этап 2):** `regularity_cost` неубывает по каждому факту
   (веса ≥ 0).
10. **Поза необратимости (этап 3):** `classify_reversible` по умолчанию
    консервативна — недоверенный источник всегда `False`; обогащение может
    только повысить права, не выдать их.
11. **Аддитивность обогащения (этап 3):** `Option.guard`/`Option.effect` —
    необязательны; без них оценка совместима с этапом 1.
12. **Реактивность BT (этап 4):** `tick` с корня каждый тик — дерево
    перерешается по текущему контексту, без «второго executive».
13. **Детерминизм BT (этап 4):** `tick` и `order_children` — чистые; одинаковый
    вход → одинаковый статус и память; тай-брейк — исходный порядок.
14. **Преемпция (этап 4):** `FALLBACK` каждый тик перепроверяет более
    приоритетных детей; бегущий лист вытесняется, а не «доигрывается вслепую».
15. **Изоляция ввода-вывода (этап 4):** исходы действий входят в Core через
    `TickContext.action_status` (инжектит Shell); `tick` не исполняет эффекторы.
16. **Выводимость дерева (этап 5):** `backward_chain` строит дерево из
    `effect`/`guard` опций; ни один узел не написан под конкретную задачу.
17. **Горизонт (этап 5):** глубина цепочки ≤ `max_depth`; исчерпание → безопасный
    отказ (`Condition`), не фиктивное действие.
18. **Детерминизм генератора (этап 5):** чистый `backward_chain`; тай-брейк —
    порядок окна опций; одинаковый вход → одинаковое дерево.
19. **Безопасный отказ (этап 5):** недостижимая цель → `Condition`-узел на цель
    (провал честен), а не выдуманный `Action`.

## Критерии приёмки

**Этап 1:**
- [ ] `Option`/`OptionWindow`/`OptionContext`/`OptionCandidate`/`OptionTrace` —
      frozen, с валидацией
- [ ] `build_options` — открытое окно из `Affordance`, стабильный порядок
- [ ] `score_option` — тотален, дефолт для неизвестного, `relevance` как обогащение
- [ ] `select_option` — детерминированный argmax + тай-брейк + полная трасса
- [ ] необратимая опция видна и занижена (не скрыта)
- [ ] `ValueError` на некорректных входах
- [ ] ruff/тесты зелёные; pyright strict для Core
- [ ] `Affordance`/`select_affordance` не ломаются (совместимость S6)

**Этап 2:**
- [ ] `Fact`/`Guard`/`Regularity` — frozen, с валидацией (`default`/`threshold`/
      `weight` в допустимом диапазоне)
- [ ] `evaluate_fact` — тотален, дефолт для неизвестного, клип в [0, 1]
- [ ] `guard_holds` — порог, чистая
- [ ] `regularity_cost` — сумма с весами, пустой вход → 0.0, монотонность
- [ ] реестр `FACTS` — единый источник имён (Core и Shell не расходятся)
- [ ] неизвестный факт деградирует к `fact.default`, не падает
- [ ] `ValueError` на некорректных входах

**Этап 3:**
- [ ] `Effect`/`Actuation`/`ActuationResult`/`ToolAnnotations` — frozen, с валидацией
- [ ] `ActuationKind`/`ActuationStatus` — enum
- [ ] `classify_reversible` — консервативный дефолт (недоверенный → `False`)
- [ ] `Option.guard`/`Option.effect` — аддитивны, дефолт `None`
- [ ] Core не импортирует `integrations` (trusted — от Shell)
- [ ] `ValueError` на некорректных входах

**Этап 4:**
- [ ] `NodeKind`/`NodeStatus` — enum; `Node` — frozen, с валидацией формы
      (лист ↔ guard/actuation, композит ↔ children)
- [ ] `tick` — чистый, тотальный, реактивный (с корня каждый тик)
- [ ] `CONDITION` — `guard_holds`; `ACTION` — статус из `TickContext`, дефолт `RUNNING`
- [ ] `SEQUENCE` — порядок, resume по `TickMemory`, `FAILURE` останавливает
- [ ] `FALLBACK` — приоритет, `SUCCESS` останавливает, преемпция бегущего
- [ ] `order_children` — устойчивая сортировка по `regularity_cost`
- [ ] детерминизм: одинаковый вход → одинаковый статус и память
- [ ] `ValueError` на некорректной форме узла

**Этап 5:**
- [ ] `Goal` — frozen, с валидацией `value ∈ [0, 1]`
- [ ] цель уже истинна → `Condition`-узел (без действия)
- [ ] опция с `guard` → `Sequence` (подцель предваряет действие)
- [ ] недостижимая цель → безопасный отказ (`Condition`, не фиктивный `Action`)
- [ ] `max_depth` обрезает цепочку (горизонт)
- [ ] детерминизм: одинаковый вход → одинаковое дерево; тай-брейк — окно
- [ ] ablation: убрать `effect` → цепочка схлопывается
- [ ] `max_depth < 1` → `ValueError`

## Границы (этапность)

**Входит (этап 1):** окно, скорер, выбор, трасса — для тулов.

**Входит (этап 2):** `Fact`/`Guard`/`Regularity`, `evaluate_fact`/`guard_holds`/
`regularity_cost`, реестр `FACTS`.

**Входит (этап 3):** `Effect`, `Actuation`/`ActuationResult`/`ActuationKind`/
`ActuationStatus`, `ToolAnnotations`/`classify_reversible`, поля
`Option.guard`/`Option.effect`.

**Входит (этап 4):** `Node`/`NodeKind`/`NodeStatus`, `TickContext`/`TickMemory`,
`tick`, `order_children` (BT: Condition/Action/Sequence/Fallback).

**Входит (этап 5):** `Goal`, `backward_chain` (runtime-генератор дерева от цели,
горизонт 2–3).

**НЕ входит (расширят спеку перед кодом):**
- **BT-декораторы и `Parallel`** — не вводим до реального кейса;
- **прокидка аннотаций MCP** (`tools/list` → `ToolInfo` → `Affordance`) — отдельный шаг;
- **встроенные речевые действия** в окне (стык с `core/policy`);
- **исполнение** (эффекторы, executor, async, ожидание) — `host/executor`;
- **привязка темы через эмбеддинги** (здесь — только поле `relevance`; эмбеддер — Shell);
- **категориальные факты** (режим и т.п.) — только градуированные ∈ [0, 1].

## Open Questions

1. **Форма обогащения `relevance`:** порог `relevance_floor` и как считать
   `uncertainty·relevance` — калибруется при реализации (ADR-0012 §10).
2. **`irreversible_penalty`:** дефолт `0.5` — стартовая величина; реальная
   авторизация — за гейтом, не в скорере. Калибровка по телеметрии.
3. **`BUILTIN`-опции:** форма pragmatic для встроенных действий (этап 2) —
   уточнить при стыке с `core/policy`.
4. **Замена `select_affordance`:** оставить как S6-fallback или удалить после
   wiring — решить на этапе wiring (`host/loop.py`).
5. **Начальный набор `FACTS` (этап 2):** какие факты заводим первыми
   (`network_available`, `topic_bound`, ...) — уточняется при связывании с
   опциями (этап 3); реестр расширяем.
