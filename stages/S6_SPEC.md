# SPEC.md — S6: Автономия

Стадия S6 из `BUILD_ROADMAP.md` (§3, §4.2). Цель — хост **работает на длинном
горизонте**: консолидирует память во сне, сам замечает тихий дрейф и
инициирует управляемый сброс, исследует неопределённость эпистемическим
драйвом и ведёт минимальный дискретный (факторизованный) слой состояний.

Предпосылки: S1–S5 завершены. Ворота S6 — `VALIDATION.md` §4. Решения —
ADR-0009 (автономия), ADR-0005 §3 (дискретный слой), ADR-0004 (FC/IS),
ADR-0007 (LLM — актюатор), ADR-0008 §5 (мягкий интент), INTENT §3–4.

## Область (входит)

1. **Самоконтроль** (`src/core/selfcontrol/`): метакогнитивные наблюдаемые,
   детектор critical slowing down, классификатор «развитие vs дрейф», протокол
   сброса (три уровня).
2. **Консолидация памяти** (`src/memory/`): чистый план pruning + Structure
   Learning (схемы); исполнение через Shell; явное логирование.
3. **Эпистемический драйв** (`src/core/policy/`): `Action.EXPLORE`; мягкий
   интент по неопределённости; `IDENTIFY_PARTNER` как частный случай.
4. **Дискретный факторизованный слой** (`src/core/factorization/`): независимые
   факторы (mode/partner/task), маргиналы, обновление.
5. **Расширение `PolicyContext`**: поле `metacognition` с дефолтом (без ломки
   вызывающих).
6. **`AutonomyConfig`** + валидация; телеметрия автономии.

## Явно НЕ входит

- **Длинный горизонт (C10) как прогон** — проход 2.
- **HITL-протокол сброса и самоотчёт** — вынесены в отдельный этап **S7**
  (ADR-0010); в S6 проходе 1 есть Core-механики (`ResetPlan`, классификатор),
  диалоговое подтверждение и самоотчёт — S7-D.
- **Ночной цикл по расписанию** (триггер сна) — проход 2; в проходе 1
  консолидация вызывается явно.
- **MCP-действия** (эпистемическое зондирование, аффордансы) — проход 2.
- **`pymdp`** — опциональное будущее; проход 1 — NumPy-факторизация.
- **Мета-пластичность / само-модель (уровень B)** — горизонт, не обязательство.
- **Четвёртая категория сигналов в шину** — проход 1 подаёт метакогницию в
  policy, не в колонки.

## Решения стадии

Зафиксированы в `adr/0009-autonomy-consolidation-selfcontrol-and-factorization.md`:

- **A. Метакогниция — поле контекста, не канал шины (проход 1).**
- **B. Триггер сброса — critical slowing down, не стресс.**
- **C. Три механизма: reset / regime shift / catastrophic drift.**
- **D. Core не сбрасывается: reset меняет установки, не якоря.**
- **E. Консолидация — явная, логируемая (инвариант 6).**
- **F. Драйв — `Action.EXPLORE`, чистая оценка; `IDENTIFY_PARTNER` — частный
  случай.**
- **G. Факторизация — NumPy, без новой зависимости.**
- **H. Порядок Core → Shell; два прохода.**

## 1. Самоконтроль (`src/core/selfcontrol/`)

### 1.1 Метакогнитивные наблюдаемые (Core)

```python
@dataclass(frozen=True)
class Metacognition:
    """Снимок метакогнитивных наблюдаемых (read-only, не дублирует аффект).

    Attributes:
        conflict: Разброс активностей колонок (несогласие ансамбля), [0, 1].
        metastability: Частота смен аттрактора в окне, [0, 1].
        epistemic_uncertainty: Неопределённость идентичности/предсказания, [0, 1].
        saturation: Насыщение/тренд F (хронизация нагрузки), [0, 1].
    """
```

```python
def compute_conflict(scores: Vector) -> float:
    """Несогласие ансамбля: нормированный разброс scores ∈ [0, 1] (чистая).

    Энтропия/коэффициент вариации по scores, нормированные в [0, 1].
    """

def compute_metastability(switch_flags: Sequence[bool], *, window: int) -> float:
    """Частота смен аттрактора в окне ∈ [0, 1] (чистая)."""

def compute_saturation(f_trend: Sequence[float]) -> float:
    """Насыщение: доля тиков выше порога/положительный тренд ∈ [0, 1] (чистая)."""
```

### 1.2 Critical slowing down (Core)

```python
@dataclass(frozen=True)
class CriticalSlowingDown:
    """Признак приближения к смене режима (Scheffer/Dakos).

    Attributes:
        variance: Дисперсия метрики в окне.
        autocorrelation: Lag-1 автокорреляция метрики в окне.
        slowing: Нормированный признак slowing down, [0, 1].
        is_warning: slowing >= порога.
    """

def critical_slowing_down(
    series: Sequence[float], *, variance_gain: float, autocorr_gain: float
) -> CriticalSlowingDown:
    """Оценить critical slowing down по окну (чистая).

    ``slowing = sigmoid(variance_gain·z_var + autocorr_gain·z_ac)``;
    ``is_warning`` — по порогу. Рост дисперсии И автокорреляции — ранний
    признак смены режима (не сам стресс).
    """
```

### 1.3 Классификатор развитие vs дрейф (Core)

```python
class ChangeKind(Enum):
    STABLE = "stable"
    DEVELOPMENT = "development"
    DRIFT = "drift"

@dataclass(frozen=True)
class ChangeAssessment:
    kind: ChangeKind
    core_preserved: bool
    traceable: bool
    coherent: bool
    reason: str

def classify_change(
    *,
    core_preserved: bool,
    traceable: bool,
    coherent: bool,
    changed: bool = True,
) -> ChangeAssessment:
    """Классифицировать изменение по трём осям INTENT §4 (чистая).

    Все три оси сохранены → DEVELOPMENT; иначе → DRIFT. ``changed=False`` →
    STABLE. Пороги/оси передаются вызывающим (Shell) — конфигурируемо.
    """
```

### 1.4 Протокол сброса (Core)

```python
class ResetLevel(Enum):
    SOFT = "soft"       # adaptive reset: сброс к baseline для поиска
    FREEZE = "freeze"   # regime shift: управляемый переход установок
    HARD = "hard"       # catastrophic drift: аварийная остановка

@dataclass(frozen=True)
class ResetPlan:
    level: ResetLevel
    reason: str
    triggered: bool

def plan_reset(
    *,
    slowing: CriticalSlowingDown,
    assessment: ChangeAssessment,
    soft_threshold: float,
    hard_core_broken: bool,
) -> ResetPlan:
    """Спланировать сброс по признаку смены режима и классификации (чистая).

    - core нарушен → HARD (никогда не «сброс к норме»);
    - DRIFT + warning → FREEZE (громкий, наблюдаемый);
    - warning без DRIFT → SOFT (adaptive resetting).
    Core не сбрасывается (ADR-0009 §2, INTENT §4).
    """
```

### 1.5 Shell (`SelfMonitor`)

```python
class SelfMonitor:
    """Shell: копит окно метрик, считает наблюдаемые, ведёт счётчик смен."""
    def observe(
        self,
        *,
        scores: Vector,
        switched: bool,
        f: float,
        partner_uncertainty: float,
    ) -> tuple[Metacognition, ResetPlan]: ...
    @property
    def metacognition(self) -> Metacognition | None: ...
```

Окна фиксированной длины (кольцевые буферы); сбой — безопасный дефолт.

## 2. Консолидация памяти (`src/memory/consolidation.py`)

```python
@dataclass(frozen=True)
class ConsolidationPlan:
    """План консолидации (чистый, до исполнения).

    Attributes:
        prune_ids: id эпизодов к удалению (низкий вес/давность).
        schema_seeds: id-представители кластеров для схем.
        schema_members: id участников каждого кластера (параллельно seeds).
        kept: сколько эпизодов сохранено.
        reason: Человекочитаемая причина плана.
    """

def plan_consolidation(
    episodes: Sequence[Episode],
    *,
    min_weight: float,
    schema_threshold: float,
    max_schemas: int,
    now: float,
) -> ConsolidationPlan:
    """Построить план pruning + Structure Learning (чистая).

    Pruning: эпизоды с весом ниже ``min_weight`` → удаление.
    Structure Learning: жадная кластеризация по косинусу (порог
    ``schema_threshold``), до ``max_schemas`` схем; схема — усреднённый
    центроид кластера.
    """

@dataclass(frozen=True)
class Schema:
    centroid: Vector
    member_count: int
    summary: str
```

**Инвариант 6:** удаление допустимо только как **явная консолидация**;
Shell логирует число удалённых и их id. `MemoryStore.delete(ids)` — новый
метод (Shell).

```python
def consolidate(
    store: SupportsConsolidate,
    *,
    min_weight: float,
    schema_threshold: float,
    max_schemas: int,
    now: float,
) -> ConsolidationResult: ...
```

## 3. Эпистемический драйв (`src/core/policy/`)

```python
class Action(Enum):
    RESPOND = "respond"
    SILENT = "silent"
    INITIATIVE = "initiative"
    IDENTIFY_PARTNER = "identify_partner"
    EXPLORE = "explore"   # S6: эпистемический драйв
```

`_evaluate_explore`: эпистемическая ценность = метакогнитивная неопределённость
(`epistemic_uncertainty`) при отсутствии сообщения и превышении порога; мягкий
интент (порог + гистерезис). `IDENTIFY_PARTNER` — частный случай (агентная
неопределённость). Порог — `Preferences.explore_threshold`.

## 4. Дискретный факторизованный слой (`src/core/factorization/`)

```python
@dataclass(frozen=True)
class Factor:
    """Независимый дискретный фактор (состояние + приор)."""
    name: str
    states: tuple[str, ...]
    prior: Vector            # нормированное распределение
    likelihood: Vector       # наблюдение → состояния (нормировано)

@dataclass(frozen=True)
class FactorizedState:
    """Совместное состояние как произведение независимых факторов."""
    factors: tuple[Factor, ...]

def posterior(factor: Factor) -> Vector:
    """Байесовское обновление одного фактора: prior ∘ likelihood → норм. (чистая)."""

def marginal(state: FactorizedState, name: str) -> Vector:
    """Маргинал фактора по имени (чистая)."""

def update_factor(state: FactorizedState, name: str, observation: Vector) -> FactorizedState:
    """Обновить один фактор по наблюдению, остальные не трогать (чистая)."""

def argmax_state(factor: Factor) -> str:
    """Наиболее вероятное состояние фактора (чистая)."""
```

Факторы: `mode` (game/cooperative/free), `partner` (unknown/known), `task`.
Совместное распределение не материализуется (sparse-факторизация, манифест
§3.Е). Обновление — независимое по факторам.

## 5. Расширение `PolicyContext`

```python
@dataclass(frozen=True)
class PolicyContext:
    ...
    partner: PartnerView | None = None              # S5
    metacognition: MetacognitionView | None = None  # S6; None → S5-совместимость
```

`MetacognitionView` — структурный Protocol (без цикла `core.policy` → `core.selfcontrol`).

## 6. Конфигурация (`AutonomyConfig`)

```python
@dataclass(frozen=True)
class AutonomyConfig:
    enabled: bool = False
    # selfcontrol
    metacog_window: int = 50
    csd_variance_gain: float = 1.0
    csd_autocorr_gain: float = 1.0
    csd_warning_threshold: float = 0.6
    reset_soft_threshold: float = 0.5
    # consolidation
    consolidate_min_weight: float = 0.1
    schema_threshold: float = 0.8
    max_schemas: int = 8
    # epistemic drive
    explore_threshold: float = 0.6
    # factorization
    factor_learning_rate: float = 0.3
```

Валидация: окна ≥ 1, пороги ∈ [0, 1], learning_rate ∈ (0, 1], gains ≥ 0.
`HostConfig` получает поле `autonomy: AutonomyConfig`.

## 7. Телеметрия (расширение)

| Поле | Тип | Смысл |
|---|---|---|
| `metacog_conflict` | float | Несогласие ансамбля |
| `metacog_metastability` | float | Частота смен аттрактора |
| `metacog_saturation` | float | Насыщение/тренд F |
| `reset_level` | str | Уровень сброса ("" если нет) |
| `change_kind` | str | stable/development/drift |
| `consolidated_pruned` | int | Сколько эпизодов удалено (0 если не было) |

Новые поля — keyword-only с дефолтами (обратная совместимость). Наблюдаемость —
VALIDATION §2.5.

## Инварианты S6

1. **Автономия не роняет контур:** ошибки selfcontrol/консолидации → лог +
   безопасный дефолт; loop жив.
2. **Консолидация явная:** каждое удаление логируется и попадает в телеметрию
   (инвариант 6 в форме «кроме явной консолидации»).
3. **Core не сбрасывается:** reset меняет установки/приоритеты, не якоря
   (INTENT §4).
4. **Триггер — не голый стресс:** сброс только по critical slowing down +
   классификации.
5. **Метакогниция без circularity:** read-only снимок; влияет на следующий
   тик/ход, не на F того же тика.
6. **Обратная совместимость:** `autonomy.enabled=False` / `metacognition=None`
   → контур S5 идентичен.
7. **Детерминизм:** при фиксированных входах — одинаковые наблюдаемые, план
   сброса, план консолидации.
8. **LLM не рассуждает:** самоконтроль и факторизация — чистая математика
   (ADR-0007).

## Критерии приёмки

См. `VALIDATION.md` §4 (ворота S6). Кратко (проход 1):

- [ ] метакогнитивные наблюдаемые считаются и не дублируют аффект
- [ ] триггер сброса — critical slowing down, не стресс
- [ ] классификатор различает развитие/дрейф по трём осям
- [ ] протокол сброса: три уровня, core не сбрасывается
- [ ] консолидация явная и логируемая (инвариант 6)
- [ ] эпистемический драйв `EXPLORE` исследует неопределённость
- [ ] факторизация: независимые факторы, маргиналы, обновление
- [ ] `autonomy.enabled=False` → контур S5 идентичен
- [ ] `ruff`/тесты зелёные; mypy strict для новых Core-модулей

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Метакогниция: канал шины или поле контекста | Решено | поле контекста; канал в шину не вводится (C10 показал насыщение conflict, circularity без пользы) |
| Триггер сброса | Решено | critical slowing down (ADR-0009 §2) |
| Механизмы шифта | Решено | reset/regime shift/catastrophic drift (ADR-0009 §2) |
| Консолидация vs инвариант 6 | Решено | явная, логируемая (ADR-0009 §4) |
| `pymdp` vs NumPy | Решено | NumPy-факторизация (ADR-0009 §6) |
| Расписание ночного сна | Открыто | проход 2; в проходе 1 — явный вызов |
| Пороги классификатора (core/трассируемость/когерентность) | Открыто | калибровка по телеметрии C10 |
| Формула CSD | Открыто | кандидат `sigmoid(z_var·g + z_ac·g)`; уточнить при реализации |
| Ночной цикл | Отложено | проход 2 |
| HITL-сброс и самоотчёт | Решено | отдельный этап S7 (ADR-0010) |
| MCP-зондирование | Отложено | проход 2 |

## Implementation Notes

1. **FC/IS:** `selfcontrol/{models,compute,monitor}.py`,
   `consolidation.py` (Core `plan_consolidation` + Shell `consolidate`),
   `factorization/{models,compute}.py` (Core). Shell-ы с DI через Protocol.
2. **Без цикла импортов:** `core.policy` не импортирует `core.selfcontrol`;
   `MetacognitionView` — структурный Protocol (как `PartnerView` в S5).
3. **Расширяемость:** `PolicyContext.metacognition` — дефолт `None`.
4. **Память:** `MemoryStore.delete(ids)` + таблица `schemas`; переиспользовать
   `cosine_similarity`, не плодить хранилищ.
5. **Речь ≠ тик:** selfcontrol обновляется на тике (read-only), сброс —
   отдельный проход; консолидация — вне тика.
6. **Тесты:** Core — чистые функции (наблюдаемые, CSD, классификатор, план
   сброса, план консолидации, факторизация, детерминизм); Shell — интеграция
   (наблюдаемые копятся, консолидация логируется, `autonomy.enabled=False` →
   S5-совместимость).
7. **Reference:** `Homeostat` (Core+Shell), `DriftDetector` (Shell), `tm`
   (Shell с DI), `MemoryRouter` (Shell с safe-default).
