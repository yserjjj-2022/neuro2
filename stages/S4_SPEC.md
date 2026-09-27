# SPEC.md — S4: Воля

Стадия S4 из `BUILD_ROADMAP.md` (§3, §4, §5). Цель — хост **выбирает
действие**: гомеостаз даёт ошибку отклонения, policy выбирает намерение с
причинной трассировкой, критический интероцептивный сигнал проходит
рефлекс-путём за один тик, а перегрузка вызывает throttle.

**Статус: ✅ реализовано.** Проход 1 (ворота) и проход 2 (γ-барьер,
capability gate, control channel, макро-контекст) завершены 2026-09-27.

Предпосылки: S1 (честные сигналы), S2 (память), S3 (голос) завершены.
Ворота S4 — в `VALIDATION.md` §4. Решения — ADR-0005 (§2 γ, §3 дискретный
слой, §6 tiers, §7 гомеостаз, §8 control, §9 гардрейлы), ADR-0006
(непрерывность), ADR-0007 (LLM — актюатор), ADR-0008 (`IntentFrame.goal`).

## Область (входит)

1. **Гомеостаз** — сетпоинты интероцептивных каналов (battery, resources;
   опц. cpu) → нормированное отклонение → сигнал для policy.
2. **Policy / Action Selection** — выбор действия (respond / silent /
   initiative / identify_partner) с прагматической и эпистемической ценностью
   и **обязательной причинной трассировкой** (explainability).
3. **Reflex-путь** — критический сигнал (`is_reflex`) вызывает throttle не
   позже следующего тика, минуя policy.
4. **Throttle** — регуляция собственных параметров (k, dt, LLM-гейт) как
   «действие» гомеостаза; аудируется.
5. **γ как пред-колоночный барьер внимания** (проход 2) — доверие каналу
   управляет прохождением входа в колонки.
6. **Минимальный макро-контекст** (активная задача / режим) как вход policy.
7. **Capability gate** (заготовка) — единая точка side-effect, tier + audit,
   fail-safe deny.
8. **Канал горячего тестирования** — минимальный (`status`, `pause`,
   `resume`, `step`).
9. **Связка policy → речь**: `IntentFrame.goal` становится выходом policy.

## Явно НЕ входит

- **Внешние действия через MCP** (поиск, запись, вызов сервисов) — S5/S6.
  На S4 policy выбирает **только речевое поведение**; throttle — внутреннее.
- **Метакогнитивный канал** (конфликт колонок, метастабильность аттрактора,
  эпистемическая неопределённость) — **перенесён в S6** (см. §Решения).
- **Корневые приоры** (когерентность, целостность, верификация, манифест
  §3.В) — операционализация абстрактных ценностей отложена в S5/S6.
- **Полный дискретный слой (pymdp / sparse-факторизация)** — S6 (ADR-0005 §3).
- **ToM, модель партнёра, Vigilance Gate** — S5.
- **Эпистемический драйв** — S6 (мягкий интент `identify_partner` —
  заготовка по ADR-0008).
- **Полный control channel** (`inject`/`set`/`snapshot`/`restore`/`freeze`/
  `kill`) — позже; `kill` покрыт SIGINT.
- **Персистентность состояния** (snapshot/restore) — отдельная подсистема.

## Решения стадии

Зафиксированы до SPEC:

- **A. Метакогниция → S6.** Канал не проектируется в S4; ADR сейчас не
  пишем (отсутствие решения — не решение). Условие безболезненного переноса:
  вход policy — **расширяемый** `PolicyContext` (frozen dataclass); в S6
  метакогнитивные наблюдаемые добавляются новым полем, не ломая код.
  Запись живёт в `BACKLOG` (S4-строка метакогниции → перенести в S6).
- **B. Policy на S4 — только речь.** Внешние side-effect откладываются.
  Capability gate — интерфейс + audit + fail-safe deny (заготовка), не
  блокирует ворота.
- **C. Гомеостаз — battery + resources (опц. cpu).** Корневые приоры — S5/S6.
- **D. Порядок: проход 1 (ворота) → проход 2 (расширение).** Сначала
  минимальный замкнутый контур, дающий все ворота S4; затем γ-барьер,
  control, gate, макро-контекст.
- **E. Control channel — минимальный** (`status`/`pause`/`resume`/`step`).

## 1. Гомеостаз (Core + Shell)

Сетпоинт — целевой диапазон канала. Отклонение нормируется в `[0, 1]`:
до `comfort` — норма (0), при `critical` — 1 (совпадает с порогом reflex
`severity ≥ 0.9`).

```python
@dataclass(frozen=True)
class Setpoint:
    """Сетпоинт интероцептивного канала.

    Attributes:
        tag: Тег источника ("battery", "resources", "cpu").
        comfort: Severity, до которой канал считается в норме.
        critical: Severity, при которой отклонение = 1.0 (и is_reflex).
        weight: Важность канала (корневой приор, S4: дефолт 1.0).
    """

    tag: str
    comfort: float = 0.5
    critical: float = 0.9
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Валидация: 0 <= comfort < critical <= 1, weight > 0."""


@dataclass(frozen=True)
class HomeostaticSignal:
    """Оценка одного канала относительно сетпоинта.

    Attributes:
        tag: Тег канала.
        severity: Сырая severity сигнала (из SignalSource).
        deviation: Нормированное отклонение, [0, 1].
        is_critical: severity >= reflex_threshold.
    """

    tag: str
    severity: float
    deviation: float
    is_critical: bool


@dataclass(frozen=True)
class HomeostasisState:
    """Снимок гомеостаза после оценки всех каналов.

    Attributes:
        signals: Оценки по каналам (в порядке сетпоинтов).
        max_deviation: Максимум deviation (вход policy).
        severity: Максимум severity (для рефлекса).
        is_critical: Есть ли критический канал.
    """

    signals: tuple[HomeostaticSignal, ...]
    max_deviation: float
    severity: float
    is_critical: bool


def setpoint_deviation(
    severity: float, comfort: float, critical: float
) -> float:
    """Нормированное отклонение severity от сетпоинта.

    Формула:
        deviation = clip((severity - comfort) / (critical - comfort), 0, 1)

    Args:
        severity: Сырая severity канала, [0, 1].
        comfort: Верхняя граница нормы.
        critical: Severity, при которой deviation == 1.0.

    Returns:
        Отклонение ∈ [0, 1].

    Raises:
        ValueError: Если critical <= comfort.
    """
```

```python
class Homeostat:
    """Shell: оценивает интероцептивные сигналы относительно сетпоинтов.

    Чистая оценка (setpoint_deviation) + владение сетпоинтами. Читает
    ``SignalSource`` из шины (severity), не мутирует их.
    """

    def __init__(
        self,
        setpoints: Sequence[Setpoint],
        reflex_threshold: float = 0.9,
    ) -> None:
        """Создать гомеостат.

        Args:
            setpoints: Сетпоинты каналов.
            reflex_threshold: Порог severity для is_critical (0.9).
        """

    def evaluate(self, signals: Sequence[SignalSource]) -> HomeostasisState:
        """Оценить сигналы шины относительно сетпоинтов.

        Каналы без сетпоинта игнорируются. Порядок результата — порядок
        сетпоинтов.

        Args:
            signals: Сигналы последнего тика (``SignalBus.last_signals``).

        Returns:
            HomeostasisState.
        """
```

## 2. Policy / Action Selection (Core)

Вход — расширяемый контекст (условие решения A). Действия — речевые
(решение B).

```python
class Action(Enum):
    """Кандидаты-действия policy (S4: только речь)."""

    RESPOND = "respond"
    SILENT = "silent"
    INITIATIVE = "initiative"
    IDENTIFY_PARTNER = "identify_partner"


@dataclass(frozen=True)
class Preferences:
    """Предпочитаемые исходы (меняются без переобучения — goal-directed test).

    Attributes:
        respond_to_messages: Предпочитаем ли отвечать на сообщение.
        initiative_f_threshold: Порог F для инициативы без сообщения.
        homeostatic_alert: Предпочитаем ли предупреждать о перегрузке.
        pragmatic_weight: Вес прагматической ценности.
        epistemic_weight: Вес эпистемической ценности.
    """

    respond_to_messages: bool = True
    initiative_f_threshold: float = 1.0
    homeostatic_alert: bool = True
    alert_deviation: float = 0.7
    silent_baseline: float = 0.5
    silent_stress_gain: float = 0.3
    pragmatic_weight: float = 1.0
    epistemic_weight: float = 0.5


@dataclass(frozen=True)
class PolicyContext:
    """Расширяемый вход policy (S6 добавляет метакогницию новым полем).

    Attributes:
        f: Свободная энергия F(t).
        valence: Валентность.
        stress: Аллостатический стресс.
        task: Активная задача/аттрактор (тег колонки).
        homeostasis: Снимок гомеостаза.
        has_new_message: Пришло ли новое сообщение оператора.
        mode: Режим хоста (game/cooperative/free; макро-контекст, проход 2).
    """

    f: float
    valence: float
    stress: float
    task: str
    homeostasis: HomeostasisState
    has_new_message: bool
    mode: str = "free"


@dataclass(frozen=True)
class PolicyCandidate:
    """Оценка одного кандидата-действия.

    Attributes:
        action: Действие.
        pragmatic: Прагматическая ценность (близость к предпочитаемому исходу).
        epistemic: Эпистемическая ценность (снижение неопределённости).
        value: Итоговая ценность (взвешенная сумма).
        reason: Причина оценки (для трассировки).
    """

    action: Action
    pragmatic: float
    epistemic: float
    value: float
    reason: str


@dataclass(frozen=True)
class PolicyTrace:
    """Причинная трассировка решения policy (explainability — инвариант).

    Attributes:
        chosen: Выбранное действие.
        reason: Причина выбора (человекочитаемая, но выведенная из расчёта).
        candidates: Все рассмотренные кандидаты с оценками.
    """

    chosen: Action
    reason: str
    candidates: tuple[PolicyCandidate, ...]


def evaluate_candidates(
    context: PolicyContext, preferences: Preferences
) -> tuple[PolicyCandidate, ...]:
    """Оценить всех кандидатов-действий (чистая функция).

    Правила (детерминированы; «захардкожена цель, не формулировка» —
    ADR-0008):

    - ``RESPOND``: прагматическая ценность высока при
      ``has_new_message`` и ``respond_to_messages``; иначе 0.
    - ``SILENT``: базовая ценность 0; растёт при высокой F/стрессе
      (беречь ресурс) и низкой коммуникативной релевантности.
    - ``INITIATIVE``: прагматическая ценность при ``f > initiative_f_threshold``
      или при ``homeostasis.max_deviation`` выше порога (предупредить) и
      ``homeostatic_alert``.
    - ``IDENTIFY_PARTNER``: эпистемическая заготовка (S4: ценность 0 без
      драйва; включается в S6).

    Args:
        context: Текущий контекст.
        preferences: Предпочитаемые исходы.

    Returns:
        Кандидаты в фиксированном порядке действий.
    """


def select_action(
    context: PolicyContext, preferences: Preferences
) -> PolicyTrace:
    """Выбрать действие детерминированно и вернуть причинную трассу.

    Тай-брейк — порядок `Action` (стабильный). Причина выбора выводится из
    оценок кандидатов, а не генерируется постфактум (манифест §3.И:
    explainability ≠ «звучит понятно»).

    Args:
        context: Текущий контекст.
        preferences: Предпочитаемые исходы.

    Returns:
        PolicyTrace с выбранным действием и всеми кандидатами.
    """
```

**goal-directed test.** Меняем `Preferences` (например,
`respond_to_messages=False` или `homeostatic_alert=False`) → при том же
входе выбирается другое действие, без переобучения. Это отличает
целенаправленное поведение от стимул-реакции (манифест §3.И).

## 3. Reflex-путь и throttle (Shell)

Критический сигнал (`is_reflex`, severity ≥ 0.9) вызывает throttle
**не позже следующего тика**, минуя policy, dwell и attractor (манифест
§3.К). Throttle меняет **собственные** параметры хоста (внутреннее,
обратимое), а не мир.

```python
@dataclass(frozen=True)
class ThrottlePlan:
    """План регуляции собственных параметров.

    Attributes:
        active: Активен ли throttle.
        k_scale: Множитель k-WTA (<= 1).
        dt_scale: Множитель шага dt (>= 1; медленнее тики).
        llm_gate: Запретить инициативную речь (дорогой вызов LLM).
        reason: Причина (тег канала + severity).
    """

    active: bool
    k_scale: float
    dt_scale: float
    llm_gate: bool
    reason: str


def plan_throttle(
    homeostasis: HomeostasisState,
    *,
    severity_threshold: float = 0.9,
    k_scale: float = 0.5,
    dt_scale: float = 2.0,
) -> ThrottlePlan:
    """Построить план throttle по гомеостазу (чистая функция).

    Args:
        homeostasis: Снимок гомеостаза.
        severity_threshold: Порог критического сигнала.
        k_scale: Во сколько раз уменьшить k.
        dt_scale: Во сколько раз увеличить dt.

    Returns:
        ThrottlePlan; ``active=False`` при отсутствии критических каналов.
    """
```

Порядок в `HostLoop.step_once` (проход 1):

```
signals   = bus.step(tick, now)              # сигналы уже прочитаны
homeo     = homeostat.evaluate(bus.last_signals)
throttle  = plan_throttle(homeo)             # рефлекс: активен в этом же тике
dt_eff    = dt * throttle.dt_scale           # применён немедленно
k_eff     = max(1, round(k * throttle.k_scale))
...
telemetry.log(..., throttle=throttle.active, homeostasis=homeo.max_deviation,
              policy_action=..., policy_reason=...)
```

Реакция в том же тике удовлетворяет «≤ 1 тик» (severity текущего тика
отражает метрики предыдущего — см. `ResourceProvider`). LLM-гейт применяет
`ChatSession`: при `throttle.llm_gate` инициатива запрещена (ответ на
сообщение сохраняется).

**Обратимость.** Throttle — регуляция, а не дрейф конфигурации: базовое `k`
запоминается при сборке loop и восстанавливается, как только сигнал
нормализуется (`active=False`). Порог активации throttle берётся из
`HomeostasisConfig.reflex_threshold` (согласован с `is_critical`).

**Обход attractor — отложен.** В проходе 1 рефлекс = throttle + обход policy
(LLM-гейт). Полный обход `column_step`/`dwell`/attractor для критического
сигнала (манифест §3.К) не реализован — занесено в BACKLOG (техдолг).

**Честная оговорка.** Mock-battery не связана с реальным потреблением,
mock-resources измеряют compute-фазу. На S4 throttle даёт **наблюдаемый
контракт** (активация + логирование + аудит) и реальный эффект на дорогой
путь (LLM-гейт). Физический эффект на латентность — по мере появления
реально тяжёлых операций (MCP/SLM, S5+).

## 4. γ как пред-колоночный барьер внимания (проход 2)

Сейчас γ — пост-фактум вес ошибок в F (`FreeEnergyCalculator`). По
ADR-0005 §2 на S4 γ включается **до** колонок: доверие каналу управляет
прохождением входа.

Дизайн (формула уточняется при реализации — Open Question):

```
u_eff,i = u_i · a(γ_i)     a: [0, γ_max] → [0, 1], монотонно
```

Низкая γ (шумный/недоверенный канал) → вход аттенюируется (барьер
внимания); высокая γ (доверенный) → проходит. Это трогает **ядро CMC**,
поэтому идёт вторым проходом, после закрытия ворот. Обратная
совместимость: `attention_gate=False` → `u_eff = u` (контур S1–S3).

Отличие от «меняем мир / меняем модель» (ADR-0005 §4): то — уровень
policy/действия; здесь — уровень восприятия (что вообще попадает в колонки).

## 5. Макро-контекст (проход 2)

Минимальный дискретный контекст без pymdp (ADR-0005 §3):

```python
@dataclass(frozen=True)
class MacroContext:
    """Минимальный макро-контекст для policy.

    Attributes:
        task: Активная задача/аттрактор.
        mode: Режим хоста (game/cooperative/free, манифест §8).
    """

    task: str
    mode: str = "free"
```

`PolicyContext` уже несёт `task` и `mode` — `MacroContext` фиксирует их
как единый источник. Полная sparse-факторизация (режим/партнёр/задача) —
S6.

## 6. Capability gate + audit (проход 2, заготовка)

Единственная точка side-effect (ADR-0005 §9, манифест §3.И). На S4 policy
не делает внешних действий, поэтому gate применяется к внутренним
«действиям» (throttle, речь) как **интерфейс + аудит**, а не как барьер
T2/HITL.

```python
class CapabilityTier(Enum):
    """Уровни возможностей (ADR-0005 §6)."""

    T0 = "observation"
    T1 = "speech"
    T2 = "hitl_action"
    T3 = "autonomous_reversible"
    T4 = "bounded_irreversible"


@dataclass(frozen=True)
class ActionRequest:
    """Запрос на действие.

    Attributes:
        name: Имя действия ("speak", "throttle", ...).
        tier: Требуемый tier.
        reversible: Обратимо ли действие.
        reason: Причина (из policy trace).
    """

    name: str
    tier: CapabilityTier
    reversible: bool
    reason: str


@dataclass(frozen=True)
class GateDecision:
    """Решение gate.

    Attributes:
        allowed: Разрешено ли действие.
        reason: Причина решения.
    """

    allowed: bool
    reason: str


class CapabilityGate:
    """Единая точка проверки действий перед side-effect.

    Fail-safe deny: неизвестный tier, таймаут HITL или отсутствие токена →
    отказ (манифест §3.И, ADR-0005 §6). На S4 — заготовка: внешних
    необратимых действий нет.
    """

    def __init__(
        self,
        max_tier: CapabilityTier = CapabilityTier.T1,
        hitl_timeout_s: float = 30.0,
    ) -> None: ...

    def request(self, req: ActionRequest) -> GateDecision:
        """Проверить запрос действия.

        Args:
            req: Запрос.

        Returns:
            GateDecision (fail-safe deny по умолчанию для tier > max_tier).
        """
```

## 7. Канал горячего тестирования (проход 2, минимальный)

`src/host/control.py` — оперативный контроль (ADR-0005 §8). Минимальный
набор (решение E): `status`, `pause`, `resume`, `step`.

```python
class ControlChannel:
    """Локальный командный интерфейс над HostLoop (Shell).

    Минимальные команды S4: status/pause/resume/step. Расширение
    (inject/set/snapshot/restore/freeze/kill) — позже.
    """

    def __init__(self, loop: HostLoop) -> None: ...

    def status(self) -> str:
        """Строка состояния (переиспользует format_status)."""

    def pause(self) -> None:
        """Остановить тики; состояние и канал управления живы (freeze)."""

    def resume(self) -> None:
        """Возобновить тики."""

    def step(self, n: int = 1) -> int:
        """Сделать ровно n тиков (пошаговое наблюдение рефлекса)."""
```

## 8. Связка policy → речь

`IntentFrame.goal` перестаёт быть константой `"respond"` и становится
выходом policy (ADR-0008). `SpeechController` получает `PolicyTrace`
(или `goal`) и передаёт в `build_intent_frame`.

- `should_speak` (S3) заменяется/оборачивается policy: решение
  speak/silent/initiative — теперь выход `select_action`.
- `IDENTIFY_PARTNER` — речевое намерение (мягкий интент, ADR-0008);
  на S4 — заготовка, активируется драйвом в S6.
- `speech.enabled=False` → контур S1/S2 идентичен; policy не вызывается.

## 9. Телеметрия (расширение)

| Поле | Тип | Смысл |
|---|---|---|
| `policy_action` | str | Выбранное действие ("" если policy не вызывалась) |
| `policy_reason` | str | Причина выбора (трассировка) |
| `throttle` | bool | Активен ли throttle на тике |
| `homeostasis` | float | max_deviation гомеостаза |

Итого **23 поля** (было 19). Наблюдаемость — VALIDATION §2.5.

## 10. Конфигурация

```python
@dataclass(frozen=True)
class HomeostasisConfig:
    """Параметры гомеостаза.

    Attributes:
        setpoints: Сетпоинты каналов (battery, resources, cpu).
        reflex_threshold: Порог severity для рефлекса (0.9).
        throttle_k_scale: Множитель k при throttle.
        throttle_dt_scale: Множитель dt при throttle.
    """

    setpoints: tuple[Setpoint, ...] = (...)
    reflex_threshold: float = 0.9
    throttle_k_scale: float = 0.5
    throttle_dt_scale: float = 2.0


@dataclass(frozen=True)
class PolicyConfig:
    """Параметры policy.

    Attributes:
        enabled: Включать ли policy (иначе S3-поведение: should_speak).
        preferences: Предпочитаемые исходы.
        mode: Режим хоста (макро-контекст).
        attention_gate: Включать ли пред-колоночную γ (проход 2).
    """

    enabled: bool = True
    preferences: Preferences = field(default_factory=Preferences)
    mode: str = "free"
    attention_gate: bool = False
```

Валидация: `reflex_threshold ∈ [0, 1]`, `throttle_k_scale ∈ (0, 1]`,
`throttle_dt_scale ≥ 1`, `mode ∈ {game, cooperative, free}`,
`comfort < critical`, `weight > 0`.

`HostConfig` получает поля `homeostasis: HomeostasisConfig` и
`policy: PolicyConfig`.

## Инварианты S4

1. **Reflex ≤ 1 тик:** критический сигнал (severity ≥ 0.9) активирует
   throttle в том же/следующем тике (VALIDATION §2.3).
2. **Explainability:** у каждого решения policy и каждого throttle есть
   причинная запись (`PolicyTrace` / `reason`); объяснение выведено из
   расчёта, не сгенерировано постфактум (манифест §3.И).
3. **Goal-directed:** смена `Preferences` меняет поведение без
   переобучения.
4. **Policy не роняет контур:** ошибки → лог + безопасный дефолт
   (`SILENT`); loop живёт.
5. **Fail-safe deny:** gate отказывает при неизвестном tier/таймауте.
6. **Обратная совместимость:** `policy.enabled=False` → поведение S3
   (`should_speak`); `attention_gate=False` → вход колонок не изменён.
7. **Внутренние действия только:** S4 не имеет внешних side-effect
   (решение B).
8. **Детерминизм:** фиксированные входы → одинаковый `PolicyTrace`
   (чистое ядро policy).

## Критерии приёмки

См. `VALIDATION.md` §4 (ворота S4). Кратко:

- [ ] **goal-directed test:** смена предпочитаемого исхода → поведение
      перестраивается без переобучения
- [ ] reflex: критический сигнал → throttle за ≤ 1 тик (телеметрия)
- [ ] explainability: у каждого решения есть причинная запись
- [ ] ресурсный throttle срабатывает при перегрузке (C6)
- [ ] capability gate: fail-safe deny (заготовка)
- [ ] `ruff`/тесты зелёные; mypy strict для `src/core/policy`,
      `src/core/homeostasis`
- [ ] `policy.enabled=False` → контур S3 идентичен

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Метакогнитивный канал | Решено | Перенесён в S6; ADR не пишем; `PolicyContext` расширяем |
| Объём действий policy | Решено | Только речь (S4); MCP — S5/S6 |
| Порядок сборки | Решено | Проход 1 (ворота) → проход 2 (расширение) |
| Гомеостаз-каналы | Решено | battery + resources (опц. cpu) |
| Корневые приоры | Отложено | S5/S6 |
| Control channel | Решено | Минимальный: status/pause/resume/step |
| Формула пред-колоночной γ `a(γ)` | Открыто | Кандидат: нормировка в (0,1) с опорной γ; уточнить при реализации (проход 2) |
| Порог `homeostatic_alert` (предупреждать) | Открыто | Связка deviation + гистерезис; калибровка по телеметрии |
| Включение `IDENTIFY_PARTNER` | Отложено | Драйв — S6 (ADR-0008) |
| Полный gate (T2/HITL) | Отложено | По мере появления внешних действий |
| Поле `uncertainty` в PolicyContext | Отложено | S6 (метакогниция) |

## Implementation Notes

1. **FC/IS:** `homeostasis.py` (setpoint_deviation — Core; Homeostat —
   Shell), `policy.py` (evaluate_candidates/select_action — Core),
   `throttle.py` (plan_throttle — Core), `control.py`/`gate.py` — Shell.
   Размещение: `src/core/homeostasis/`, `src/core/policy/` (или
   `src/host/` для throttle/gate/control — решается в PLAN).
2. **Расширяемость контекста:** `PolicyContext` — frozen dataclass; в S6
   метакогниция добавляется полем с дефолтом (не ломает вызывающих).
3. **Порядок в loop:** гомеостаз → throttle (рефлекс) → policy → pipeline.
   Throttle влияет на dt/k текущего тика; policy — на речь (вне тика).
4. **Речь ≠ тик:** `select_action` вызывается в `ChatSession` (как
   `SpeechController.respond` в S3); медленный LLM не блокирует тики.
5. **Аудит:** throttle и решения policy логируются с причиной
   (`PolicyTrace.reason`, `ThrottlePlan.reason`).
6. **Тесты:** Core — чистые функции (детерминизм, goal-directed смена
   preferences, setpoint-границы); Shell — loop-интеграция (рефлекс ≤ 1
   тик, C6 throttle, обратная совместимость).
7. **Reference:** `PrecisionEstimator`/`DriftDetector` (energy) — образец
   Shell с состоянием; `SpeechController` — образец событийного Shell;
   `TaskAttractor` — образец FC/IS Core+Shell.
