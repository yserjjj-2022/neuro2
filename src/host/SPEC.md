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
