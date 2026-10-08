# SPEC.md — src/host

## Назначение

Host-слой: сенсорная шина, per-tick конвейер и host loop. Оживляет модули
во времени и владеет телеметрией (S1).

Компоненты:
- `sources.py` — провайдеры сигналов → `u(t)`, карта сегментов, `tags_above_threshold`.
- `wiring.py` — чистая композиция `CMCPipeline` → `TickOutcome` (без I/O).
- `resources.py` — `ResourceMeter` + `ResourceProvider` (интероцепция ресурсов).
- `throttle.py` — `ThrottlePlan` + `plan_throttle` (рефлекс-throttle, S4).
- `gate.py` — `CapabilityGate` + tiers + `Capability`-флаги (единая точка
  side-effect, S4; гранулярные права — S4-долг).
- `control.py` — `ControlChannel` (status/pause/resume/step, S4).
- `loop.py` — `HostLoop`: время, precision, гомеостаз, throttle, attention,
  ресурсы, guard, drift, policy-контекст, телеметрия.

См. ADR-0006 (временные шкалы), `stages/S1_SPEC.md`, `stages/S4_SPEC.md`.

## Поток одного тика (S4)

```
dt, now = time source (synthetic: tick·tick_dt·time_scale; wall: measured)
u_base  = SignalBus.step(tick, now)
# S4 рефлекс: критический интеро-сигнал → throttle в этом же тике
homeo   = homeostat.evaluate(bus.last_signals)     # S4
throttle= plan_throttle(homeo)                     # S4
if throttle.active: dt *= dt_scale; voting.set_k(...)  # S4
# escape hatch: streak удержания throttle → право голоса шаблоном (S4-долг)
text    = message_provider.text_at(tick)           # S2
if text != last_text:
    query = memory.context_embedding(text)         # S2 (event-triggered)
    prior = memory.recall_prior(query)             # S2
u       = concat(u_base, prior)                    # S2
γ       = PrecisionEstimator.update(u)  (variance) или ones (baseline)
if attention_gate: u_eff = u · a(γ)                # S4: пред-колоночный барьер
outcome = pipeline.tick(u_eff, γ, dt, segments, reflex_tags)
check_finite(outcome.result)            # HostIntegrityError при NaN/inf
drift   = DriftDetector.update(outcome.result)
stored  = memory.maybe_store(...)       # S2: значимое событие → эпизод
telemetry.log(..., throttle, homeostasis, policy_action, policy_reason,
              escape_hatch)
```

**Рефлекс ≤ 1 тик** (S4): `homeostat.evaluate` читает сигналы текущего тика
(severity отражает метрики предыдущего — `ResourceProvider`), throttle
применяется немедленно, до `pipeline.tick` (манифест §3.К, VALIDATION §2.3).
Throttle обратим: базовое `k` восстанавливается при `active=False`; порог
берётся из `HomeostasisConfig.reflex_threshold`.

**Пред-колоночный барьер** (S4 проход 2): при `attention_gate=True` вход
аттенюируется весами `a(γ)` (`cmc/attention.py`) — доверие каналу управляет
прохождением. `attention_gate=False` → `u_eff == u` (S1–S3).

**Policy-контекст** (S4): `loop.policy_context(...)` собирает `PolicyContext`
(через `MacroContext`: task + mode) для речевого решения вне тика;
`loop.record_policy(trace)` фиксирует решение в телеметрию следующего тика
(explainability). S6 добавляет `PolicyContext.metacognition` (снимок
`SelfMonitor`) — read-only, `None` → S5-совместимость.

**Автономия** (S6): при `autonomy.enabled=True` loop владеет `SelfMonitor`
(`selfcontrol`), обновляет наблюдаемые на каждом тике (не влияя на F того же
тика) и логирует `metacog_*`/`reset_level`/`change_kind`. `consolidate_memory()`
выполняет явную консолидацию памяти (pruning + схемы) и логирует число
удалённых (инвариант 6). `autonomy.enabled=False` → контур S5 идентичен.

**Escape hatch** (S4-долг): `loop.escape_hatch_active` истинно, когда throttle
удерживается ≥ `HomeostasisConfig.escape_hatch_ticks` тиков (streak, сброс при
норме). Даёт право сообщить о перегрузке дешёвым шаблоном без дорогого
LLM-вызова (recognition heuristic, манифест §3.Е); флаг логируется
(`escape_hatch`). Отдельно от `throttle.llm_gate`: инициатива запрещена,
escape hatch разрешён. `escape_hatch_ticks=0` → выключено.

**Коммуникативный вход — событийный** (манифест §3.Е, ADR-0006): тик —
непрерывный аффективный контур (циркадное, батарея, ресурсы), а текст
обрабатывается **один раз при появлении**. Между сообщениями сеть/recall не
трогаются; хранится последний приор (нули до первого сообщения). Приор
держится до следующего сообщения (затухание веса — BACKLOG).

`memory=None` / `MemoryConfig(enabled=False)` → контур S1 (prior не
добавляется, `total_dim == bus_dim`). Сегмент памяти:
`BusSegment("memory", offset=bus_dim, dim=prior_dim, period=1)`.

## Провайдеры (sources.py)

| Провайдер | Категория | dim | Поведение |
|---|---|---|---|
| `CircadianProvider` | extero | 2 | [sin, cos] фазы суток из `now` |
| `BatteryProvider` | intero | 1 | линейный разряд, severity=1−level |
| `CpuProvider` | intero | 1 | seeded random walk |
| `UserMessageProvider` | communicative | embedding_dim | фиксированный вектор |
| `ResourceProvider` | intero | 2 | [latency_norm, rss_norm] (S1) |
| `ConstantProvider` | extero | len(value) | константа (тест) |
| `StepProvider` | extero | len(before) | скачок (тест) |
| `NoisyProvider` | extero | dim | seeded шум (тест) |

`default_providers(resource_provider=...)` → circadian+battery+cpu+message+resources
→ ширина шины **14**.

`tags_above_threshold(errors, segments, threshold)` — чистая функция:
теги сегментов с агрегированной по колонкам ‖e‖² выше порога.

## SignalBus

```python
class SignalBus:
    def __init__(self, providers: list[SignalProvider]) -> None: ...
    def step(self, tick, now) -> Vector: ...
    @property
    def segments(self) -> tuple[BusSegment, ...]: ...
    @property
    def bus_dim(self) -> int: ...
    @property
    def last_signals(self) -> list[SignalSource]: ...
```

Инварианты: уникальные теги, `dim>0`, `period>=1`; медленный такт (period>1
кэшируется); конкатенация через `SignalRegistry.aggregate()`.

## CMCPipeline (wiring.py, без I/O)

```python
@dataclass(frozen=True)
class TickOutcome:
    result: FreeEnergyResult
    active_tags: tuple[str, ...]
    reflex_tags: tuple[str, ...]


@dataclass(frozen=True)
class CMCPipeline:
    ensemble: CMCEnsemble
    voting: VotingManager
    attractor: TaskAttractor
    observer: EnergyObserver
    active_threshold: float = 1e-8

    def tick(self, u, precision, dt, segments=(), reflex_tags=()) -> TickOutcome: ...
```

## ResourceMeter / ResourceProvider (resources.py)

```python
class ResourceMeter:
    def record_tick(self, latency_s: float) -> None: ...
    @property
    def last_latency_s(self) -> float: ...
    @property
    def last_rss_mb(self) -> float: ...
    @staticmethod
    def current_rss_mb() -> float: ...


@dataclass(frozen=True)
class ResourceProvider:
    meter: ResourceMeter
    tick_budget_ms: float = 50.0
    rss_budget_mb: float = 1024.0
    ...
```

`severity = max(latency_norm, rss_norm)` клип [0,1]; ≥0.9 → is_reflex.
**Недетерминирован** (реальный RSS) — инъекция meter для replay.

## HostLoop (loop.py)

```python
@dataclass
class HostLoop:
    bus, pipeline, logger, estimator, meter, drift
    homeostat: Homeostat | None = None          # S4
    throttle_k_scale: float = 0.5               # S4
    throttle_dt_scale: float = 2.0              # S4
    escape_hatch_ticks: int = 3                 # S4-долг
    policy_config: PolicyConfig | None = None   # S4
    tick_dt: float = 0.01
    clock_mode: str = "synthetic"
    paced: bool = False
    precision_mode: str = "variance"
    time_scale: float = 1.0

    def precision(self, u) -> Vector: ...
    def step_once(self, tick) -> TickOutcome: ...
    def run(self, max_ticks) -> int: ...
    def close(self) -> None: ...
    # S4/S5/S6:
    def policy_context(self, *, has_new_message, mode="free",
                       partner=None) -> PolicyContext: ...
    def active_task(self) -> str: ...
    def record_policy(self, trace: PolicyTrace) -> None: ...
    def record_social(self, *, trust=0.0, uncertainty=0.0, name="",
                      pause_s=0.0, claim_conflict=0.0) -> None: ...  # S5
    def consolidate_memory(self) -> int: ...                         # S6
    def record_consolidation(self, pruned: int) -> None: ...         # S6
    def explore(self, *, reason="epistemic drive") -> ProbeResult | None: ...  # S6 п2
    @property
    def escape_hatch_active(self) -> bool: ...
```

`probe_effector: ProbeEffector | None` — создаётся при `config.autonomy.enabled`
(карта аффордансов + gate). `explore()` — мягкий драйв: зондирует только при
неопределённости выше `autonomy.explore_threshold` и наличии обратимого
аффорданса; результат идёт в телеметрию следующего тика.

## ProbeEffector (probe.py, S6 проход 2)

```python
class ProbeEffector:
    def __init__(self, *, affordances=None, gate=None, probe_fn=None) -> None: ...
    def probe(self, request: ProbeRequest, *, hitl_token=None) -> ProbeResult: ...
```

Shell: исполняет эпистемическое зондирование через `CapabilityGate` (ADR-0005
§9, fail-safe deny). Обратимый аффорданс → T3/`ACT_REVERSIBLE`; необратимый →
T4/`ACT_IRREVERSIBLE` (нужен HITL-токен). Неизвестный аффорданс, отказ gate и
сбой транспорта → `ProbeResult(success=False)` без исключений (не роняет тик).
Транспорт инъецируется (`probe_fn`); по умолчанию — детерминированный mock.

`build_host_loop(config, meter=None)` — колонки под фактический `bus_dim`,
гомеостат из `config.homeostasis`, `policy_config` из `config.policy`.

## CapabilityGate (gate.py, S4 + гранулярные права)

```python
class CapabilityTier(Enum):
    T0 = "observation"; T1 = "speech"; T2 = "hitl_action"
    T3 = "autonomous_reversible"; T4 = "bounded_irreversible"


class Capability(Enum):
    READ; THINK; SPEAK; ACT_REVERSIBLE; ACT_IRREVERSIBLE


class CapabilityGate:
    def __init__(self, max_tier=CapabilityTier.T1, *, granted=None) -> None: ...
    def request(self, req: ActionRequest) -> GateDecision: ...
    def grant(self, capability: Capability) -> None: ...
    def revoke(self, capability: Capability) -> None: ...
```

Единая точка side-effect (ADR-0005 §9). Проверки по порядку: требуемые
`capabilities ⊆ granted` → tier ≤ max_tier → необратимое требует HITL-токена.
Fail-safe deny в каждом случае. Гранулярные права (S4-долг) позволяют, напр.,
под throttle сохранить `SPEAK`, отозвав `THINK` (дорогой инициативный LLM).
Речь (T1, обратимая) проходит через gate для аудита; внешних необратимых
действий на S4 нет.

## ControlChannel (control.py, S4 минимальный)

```python
@dataclass
class ControlChannel:
    loop: HostLoop
    paused: bool = False
    tick: int = 0
    def status(self) -> str: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def step(self, n=1) -> int: ...
    def run(self, max_ticks=0) -> int: ...
```

Оперативный контроль (ADR-0005 §8). `step` работает и на паузе (ручное
наблюдение рефлекса); `status` переиспользует `format_status`. Расширение
(inject/set/snapshot/restore/freeze/kill) — позже.

## Fingerprint (fingerprint.py, S7-A)

Компактный числовой отпечаток прогона (VALIDATION §5), вынесенный в Core из
тестов (ADR-0010 §3, §7). Вход — строки телеметрии (`Mapping[str, Any]`, как в
JSONL), что делает отпечаток независимым от dataclass и удобным для replay.

```python
@dataclass(frozen=True)
class FProfile:
    mean: float
    std: float
    max: float


@dataclass(frozen=True)
class BehavioralFingerprint:
    f_profile: FProfile
    reflex_count: int
    stress_peaks: float
    active_fraction: float   # доля тиков с активной колонкой, [0, 1]
    resource_alarms: int
    talk_rate: float         # доля тиков с репликой, [0, 1]
    throttle_rate: float
    explore_rate: float
    initiative_rate: float

    def metric(self, name: str) -> float: ...


def behavioral_fingerprint(events) -> BehavioralFingerprint: ...
def fingerprint_distance(a, b) -> float: ...
def regression_fingerprint(events, valence_significance=1.0) -> dict[str, float]: ...
```

`behavioral_fingerprint` (чистая) собирает отпечаток; пустой вход или
отсутствие обязательного поля → `ValueError`. `fingerprint_distance` (чистая) —
среднее нормированное расстояние в [0, 1): 0 при равенстве, симметрично.
`regression_fingerprint` — историческая подробная сводка поведенческого
регресса, перенесена из тестов без изменения поведения (тест-хелпер
реэкспортирует её под прежним именем).

## Sensitivity harness (sensitivity.py, S7-A)

Формальный тест чувствительности к ручкам (ADR-0010 §3): матрица возмущений +
проверка **инвариантов направления**, а не точных значений (точные — калибровка,
BACKLOG `[S4][policy]`). Два входа: pytest-гейт (`test_sensitivity.py`) и CLI
`--sensitivity` (ADR-0010 §7).

```python
@dataclass(frozen=True)
class SensitivityCase:
    knob: str
    values: tuple[float, ...]
    metric: str        # имя метрики BehavioralFingerprint
    direction: str     # "nondecreasing" | "nonincreasing" | "bounded"


def build_sensitivity_matrix() -> tuple[SensitivityCase, ...]: ...
def check_direction(metric_values, *, direction, tol=1e-9) -> bool: ...


class SensitivityRunner:
    def __init__(self, *, ticks=120, workdir=None) -> None: ...
    def run_case(self, case, *, seed, ticks=None) -> tuple[float, ...]: ...
    def run_all(self, *, seed, ticks=None) -> list[SensitivityResult]: ...
```

`SensitivityRunner` (Shell) прогоняет `HostLoop` при каждом значении ручки,
ведя policy вручную (`select_action` + `record_policy` + `mark_spoke`, без LLM,
ADR-0007), и берёт метрику из `BehavioralFingerprint`. Прогоны детерминированы
(synthetic, fake embedder, детерминированный meter, фиксированный seed) →
одинаковый seed даёт одинаковые отпечатки. Хардкодятся ручки и диапазоны;
проверяются инварианты направления.

## Diagnostic session (diagnostic.py, probes.py, S7-C)

Диагностическая сессия (ADR-0010 §5): проба → числовой снимок → категориальный
вердикт → ветвление → журнал. **Ручки заморожены пресетом** (сессия не меняет
конфиг), вердикт — категория (не число), снимок привязан к вердикту и не
оценивается человеком. I/O инъектируется (`input_fn`/`output_fn`), журнал —
JSONL.

```python
class Verdict(Enum): MATCHES = "matches"; PARTIAL = "partial"; MISMATCH = "mismatch"
def parse_verdict(text) -> Verdict: ...
def next_probe(probe, verdict) -> str | None: ...

@dataclass(frozen=True)
class ProbeSetup: seed: int; ticks: int; messages: tuple[tuple[int, str], ...]
@dataclass(frozen=True)
class Probe: id; stage; preset; setup; question; branches; fallback
@dataclass(frozen=True)
class DiagnosticSnapshot: tick; f; valence; stress; gamma; task;
    partner_trust; partner_uncertainty; metacog_conflict; reset_level; change_kind
@dataclass(frozen=True)
class ProbeResult: probe_id; verdict; comment; snapshot; next_probe

def take_snapshot(loop) -> DiagnosticSnapshot: ...

class DiagnosticSession:
    def __init__(self, *, probes, input_fn=input, output_fn=print,
                 journal_path=None, workdir=Path(".diagnostic"),
                 loop_factory=...) -> None: ...
    def snapshot(self) -> DiagnosticSnapshot: ...
    def run_probe(self, probe) -> ProbeResult: ...
    def run(self, *, start, max_probes=0) -> list[ProbeResult]: ...
```

`probes.py` — реестр `id → Probe`: встроенное дерево S3–S6 с ветвлениями и
fallback (`default_probes`/`default_start`) и JSON-загрузчик (`load_probes`).
Порядок проб — по возрастанию стоимости (сначала дешёвые tone/attention).
`ControlChannel.snapshot()` переиспользует `take_snapshot` (единый источник
снимка). CLI: `--diagnose [--probes FILE] [--diagnose-start ID]
[--diagnose-log PATH]`.

## Behavioral chain (behavioral_chain.py, VALIDATION §7)

Поведенческий автотест хоста: проверяет **канал состояния** по звеньям, а не
«правильные ответы». Единица теста — звено; цепочка — композиция звеньев, чтобы
провал локализовался. Блок тестов S7 (отдельной стадии нет).

```python
class ReactionClass(Enum): RESPOND; SILENT; INITIATIVE; IDENTIFY_PARTNER; EXPLORE; ESCAPE_HATCH
class StateInvariant(Enum): FINITE; F_NONNEG; STRESS_NONNEG; PARTNER_BOUNDED
class IntentInvariant(Enum): GOAL_CONSISTENT; AFFECT_CONSISTENT; TASK_CONSISTENT
class ActuationInvariant(Enum): LLM_CALLED_IFF_SPEAK; RESPONSE_RETURNED; FRAME_GROUNDED
class ReplyInvariant(Enum): CLASS_MATCHES_GOAL; NONEMPTY
class ReplyClass(Enum): STATEMENT; QUESTION; EMPTY
class PreconditionKind(Enum): BORN; PRIMED; MATURED

@dataclass(frozen=True)
class Precondition:  # VALIDATION §7.8
    kind: PreconditionKind; warmup: int
    @classmethod
    def born(cls) -> Precondition: ...
    @classmethod
    def primed(cls, warmup: int) -> Precondition: ...
    @classmethod
    def matured(cls) -> Precondition: ...
    def applies_to(self, run: Precondition) -> bool: ...

@dataclass(frozen=True)
class StateView: tick; f; valence; stress; gamma; task; active_columns; drift;
    partner_trust; partner_uncertainty
@dataclass(frozen=True)
class IntentView: frame; action; state_valence; state_stress; state_task
@dataclass(frozen=True)
class ActuationView: decision; called; response; frame_goal
@dataclass(frozen=True)
class ReplyView: frame; text; reply_class

def classify_reaction(trace, *, escape_hatch=False) -> ReactionClass: ...
def check_state(view, invariants) -> tuple[str, ...]: ...
def check_intent(view, invariants) -> tuple[str, ...]: ...
def check_actuation(view, invariants) -> tuple[str, ...]: ...
def check_reply(view, invariants) -> tuple[str, ...]: ...
def intent_from_state(view, action) -> IntentView: ...
def classify_reply(text, frame) -> str: ...

@dataclass(frozen=True)
class Scenario: id; link; preset; seed; ticks; messages; expect_reaction;
    state_invariants; intent_invariants; actuation_invariants; reply_invariants;
    precondition
@dataclass(frozen=True)
class ScenarioResult: scenario; reactions; violations; intent_violations;
    actuation_violations; reply_violations; llm_calls; passed; reason;
    throttled_calls

def default_scenarios() -> tuple[Scenario, ...]: ...
def summarize(results) -> Mapping[str, object]: ...
def report_dict(results) -> dict[str, Any]: ...  # JSON-отчёт (summary + scenarios)
def operation_facts(result) -> tuple[OperationFact, ...]: ...  # декомпозиция §7.1
def observed_shares(result) -> dict[str, float]: ...  # доли наблюдаемых (без expected)
def baseline_dict(results) -> dict[str, Any]: ...  # JSON-эталон наблюдаемых долей
def compare_to_baseline(results, baseline, *, band=0.05) -> tuple[Deviation, ...]: ...

@dataclass(frozen=True)
class OperationFact: operation; count; reason; expected  # holds: count == expected

@dataclass(frozen=True)
class Deviation: scenario_id; operation; observed; baseline; band  # delta/within_band

class BehavioralChainRunner:
    def __init__(self, *, workdir=None, llm=None) -> None: ...
    def run(self, scenario) -> ScenarioResult: ...
    def run_all(self, scenarios=None, *, precondition=None) -> list[ScenarioResult]: ...
    def run_matured(self, scenarios=None) -> list[ScenarioResult]: ...
    def run_ablation(self, check) -> AblationResult: ...
    def run_ablations(self, checks=None) -> list[AblationResult]: ...

@dataclass(frozen=True)
class RecordingLlmClient:  # delegates to FakeLlmClient, records calls
    def reply(self, messages, max_tokens=256) -> str: ...
    @property
    def calls(self) -> tuple[list[dict], ...]: ...
```

`BehavioralChainRunner` (Shell) прогоняет `HostLoop` с детерминированными
ручками (preset + synthetic clock + fake embedder + `DeterministicMeter`),
ведёт policy вручную (без LLM — ADR-0007). Звено 4: при решении «говорить»
вызывается `SpeechController.respond(decision=...)` через `RecordingLlmClient`
(LLM вызван только тогда); звено 5: `classify_reply` извлекает класс ответа и
сверяется с `frame.goal`. При `policy.enabled=False` решения не принимаются
(ablation). Абсолютные значения не проверяются — только инварианты.

**Предусловия (VALIDATION §7.8).** Сценарий декларирует объём истории:
`Precondition.born()` (с нуля — ворота), `Precondition.primed(n)` (прогрев N
сообщений в том же прогоне) и `Precondition.matured()` (длинный прогон,
`long-horizon`). Для `primed(n)` первые N сообщений — прогрев: loop
прогоняется (история копится), но наблюдаемые не записываются, замер
начинается с тика следующего сообщения. `run_all(precondition=...)` оставляет
только применимые сценарии (`Precondition.applies_to`), `run_matured` — вход
длинного прогона. Предусловие — **формат прогона**, а не объект проверки:
`born/primed` проверяют исправность канала, накопление (узнавание/recall/дрейф)
— накопительный harness mature-уровня (§7.8).

Два входа, как у sensitivity: pytest-гейт (`test_behavioral_chain.py`) и CLI
`--behavioral [--behavioral-precondition born|primed|matured]
[--behavioral-warmup N] [--behavioral-json PATH] [--behavioral-baseline PATH]
[--behavioral-save-baseline PATH] [--behavioral-band F]` (ADR-0010 §7). CLI
печатает **микроотчёт** по сценарию: заголовок (id · звено · предусловие —
вердикт) и операции из `operation_facts` (число, доля от измеренных тиков,
причина, ожидаемое с ✓/✗); итог `summarize`. При `--behavioral-json` пишет
`report_dict` (сериализуемый `{"summary", "scenarios"}`); код выхода 0 без
провалов, иначе 1.

**Декомпозиция чисел (VALIDATION §7.1).** `operation_facts` разлагает прогон на
операции и делает числа интерпретируемыми: `llm_calls` выводится из числа
говорящих решений за вычетом `throttled_calls` (llm_gate), escape hatch —
отдельная операция (LLM не вызывается) и в говорящие не входит. Инварианты
показывают `expected` (точно выведенное значение), наблюдаемые — только число и
долю (калибруются по эталону, не проверяются). Пример строки:
`вызвал LLM: 125 раз (8%) — ожидаемо 125 ✓ (= 127 говорящих решений − 2 throttle)`.

**Эталон наблюдаемых (калибровка, §7.1).** `--behavioral-save-baseline PATH`
пишет снимок долей (`baseline_dict`), `--behavioral-baseline PATH` читает его и
дополняет наблюдаемые строки смещением: `эталон 42% (Δ+3%) ✓`. Полоса —
`--behavioral-band` (по умолчанию ±5%). `compare_to_baseline` сравнивает
**объединение** операций (отсутствующая = доля 0.0), ловя и появление, и
исчезновение поведения. Выход за полосу — калибровочный сигнал (§7.7), а не
провал: код выхода от него не зависит.

## Fidelity harness (behavioral_chain.py, VALIDATION §7.6)

Проверяет **преобразователь** (LLM), а не хост: сохранение сигнала, а не
«нейтральность». Абсолютные значения не проверяемы — проверяется **порядок**.

```python
class ToneAxis(Enum): VALENCE; STRESS; GOAL_SCOPE
class DistortionClass(Enum): NONE; MASKING; INVERSION; FABRICATION
class ToneScorer(Protocol):
    def score(self, text: str, axis: ToneAxis) -> float: ...

@dataclass(frozen=True)
class LexiconToneScorer:  # deterministic polarity lexicon (CI baseline)
    def score(self, text, axis) -> float: ...

@dataclass(frozen=True)
class EmbeddingToneScorer:  # opt-in: anchors + Embedder
    embedder: Embedder
    def score(self, text, axis) -> float: ...

@dataclass(frozen=True)
class FidelityPair: id; frame_a; frame_b; axis; expect_ordered
@dataclass(frozen=True)
class FidelityResult: pair; score_a; score_b; ordered; distortion; passed; reason

def check_fidelity(pair, *, scorer) -> FidelityResult: ...

class FidelityHarness:
    def __init__(self, responder, *, scorer=None) -> None: ...
    def run(self, pair) -> FidelityResult: ...
    def run_all(self, pairs=None) -> list[FidelityResult]: ...
```

Классы искажения: маскирование (сигнал не читается), инверсия (знак
перевёрнут), фабрикация (сигнал добавлен). CI — детерминированный responder +
`LexiconToneScorer`; реальная LLM/эмбеддеры — opt-in. LLM-as-judge для ворот не
используется (ADR-0007).

## Инварианты

1. `dt > 0`; decay/интегралы в секундах.
2. `synthetic` + детерминированные каналы → replay; ресурсы — исключение.
3. `time_scale` масштабирует субъективное время (1.0 = жизнь).
4. Непрерывность: эмоциональный контур всегда включён (ADR-0006).
5. Non-finite → HostIntegrityError (fail-fast).
6. Телеметрия: 24 плоских поля (S4 + escape hatch).
7. **Рефлекс ≤ 1 тик:** критический сигнал → throttle в том же тике.
8. **Обратимость:** throttle восстанавливает базовое `k` при `active=False`;
   escape hatch сбрасывается при нормализации сигнала.
9. **Обратная совместимость:** `homeostat=None` → throttle неактивен;
   `policy_config=None`/`attention_gate=False` → контур S1–S3;
   `escape_hatch_ticks=0` → escape hatch выключен.

## Критерии приёмки (S1)

- [x] шина конкатенирует, карта сегментов непрерывна
- [x] медленный такт (period)
- [x] `tags_above_threshold` — чистая
- [x] `TickOutcome` без I/O; loop логирует
- [x] C1–C6 проходят; регресс-фингерпринт
- [x] synthetic детерминирован (с fake meter)

## Явно НЕ входит

- Реальные интеграции (datetime/psutil/MCP) — позже
- Проекции колонок (reads) — S2+
- Внешние действия / capability gate в полном виде (T2/HITL) — S5/S6
- Полный control channel (inject/set/snapshot/restore/freeze/kill) — позже
- Полный обход attractor/dwell рефлексом — осознанный техдолг (решение
  2026-10-04, BACKLOG `[S4][reflex]`): throttle + LLM-гейт дают наблюдаемый
  контракт; полный обход ядра — при реально тяжёлых критических операциях

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Расположение провайдеров | Решено | `src/host/sources.py` |
| Карта сегментов | Решено | заложена сразу |
| Инъекция meter | Решено | `build_host_loop(config, meter)` |
| Частота heartbeat | Решено | 10 Гц (dt=0.1) |
