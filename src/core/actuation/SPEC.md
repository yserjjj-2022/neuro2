# SPEC.md — src/core/actuation

## Назначение

Ядро **секвенирования актуаций** (стадия S8, ADR-0012). Модуль отвечает на
вопрос «**какие возможности есть и какая из них ценнее сейчас**» — открытое окно
опций и тотальный скорер. Это фундамент спринта: BT/async/генератор —
надстройка над правильным выбором (`stages/ACTUATION_PLAN.md`).

Спека покрывает **первый вертикальный срез** (этап 1): открытое окно опций,
оценка и выбор — заменяющие нынешнее вырожденное правило
`select_affordance` («первый обратимый», `src/mcp/probe.py`). Последующие этапы
(факты/`Guard`/`Regularity`, эффекты, BT, генератор) **расширят эту спеку** перед
своим кодом — здесь они не описываются (см. «Границы»).

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

## Критерии приёмки

- [ ] `Option`/`OptionWindow`/`OptionContext`/`OptionCandidate`/`OptionTrace` —
      frozen, с валидацией
- [ ] `build_options` — открытое окно из `Affordance`, стабильный порядок
- [ ] `score_option` — тотален, дефолт для неизвестного, `relevance` как обогащение
- [ ] `select_option` — детерминированный argmax + тай-брейк + полная трасса
- [ ] необратимая опция видна и занижена (не скрыта)
- [ ] `ValueError` на некорректных входах
- [ ] ruff/тесты зелёные; pyright strict для Core
- [ ] `Affordance`/`select_affordance` не ломаются (совместимость S6)

## Границы (этапность)

**Входит (этап 1, эта спека):** окно, скорер, выбор, трасса — для тулов.

**НЕ входит (расширят спеку перед кодом):**
- **этап 2+:** `Fact`/`Guard`/`Regularity` (кондишены «дефолт + обогащение»);
- **этап 3+:** `Effect`, `Actuation`/`ActuationResult` (единый контракт статуса);
- **этап 4+:** `Node`/`Sequence`/`Fallback` (BT), `backward_chain` (генератор);
- **встроенные речевые действия** в окне (стык с `core/policy`);
- **исполнение** (эффекторы, executor, async, ожидание) — `host/executor`;
- **привязка темы через эмбеддинги** (здесь — только поле `relevance`; эмбеддер — Shell).

## Open Questions

1. **Форма обогащения `relevance`:** порог `relevance_floor` и как считать
   `uncertainty·relevance` — калибруется при реализации (ADR-0012 §10).
2. **`irreversible_penalty`:** дефолт `0.5` — стартовая величина; реальная
   авторизация — за гейтом, не в скорере. Калибровка по телеметрии.
3. **`BUILTIN`-опции:** форма pragmatic для встроенных действий (этап 2) —
   уточнить при стыке с `core/policy`.
4. **Замена `select_affordance`:** оставить как S6-fallback или удалить после
   wiring — решить на этапе wiring (`host/loop.py`).
