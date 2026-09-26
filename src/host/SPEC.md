# SPEC.md — src/host

## Назначение

Host-слой: сенсорная шина, per-tick конвейер и host loop. Оживляет модули
во времени и владеет телеметрией (S1).

Компоненты:
- `sources.py` — провайдеры сигналов → `u(t)`, карта сегментов, `tags_above_threshold`.
- `wiring.py` — чистая композиция `CMCPipeline` → `TickOutcome` (без I/O).
- `resources.py` — `ResourceMeter` + `ResourceProvider` (интероцепция ресурсов).
- `loop.py` — `HostLoop`: время, precision, ресурсы, guard, drift, телеметрия.

См. ADR-0006 (временные шкалы), `stages/S1_SPEC.md`.

## Поток одного тика (S2)

```
dt, now = time source (synthetic: tick·tick_dt·time_scale; wall: measured)
u_base  = SignalBus.step(tick, now)
text    = message_provider.text_at(tick)      # S2: коммуникативный вход
query   = memory.context_embedding(text)       # S2 (кэш)
prior   = memory.recall_prior(query)           # S2
u       = concat(u_base, prior)                # S2: total_dim = bus_dim + prior_dim
γ       = PrecisionEstimator.update(u)  (variance) или ones (baseline)
outcome = pipeline.tick(u, γ, dt, segments, reflex_tags)
check_finite(outcome.result)            # HostIntegrityError при NaN/inf
drift   = DriftDetector.update(outcome.result)
stored  = memory.maybe_store(...)       # S2: значимое событие → эпизод
telemetry.log(...)                      # loop владеет writer'ом
```

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
    tick_dt: float = 0.01
    clock_mode: str = "synthetic"
    paced: bool = False
    precision_mode: str = "variance"
    time_scale: float = 1.0

    def precision(self, u) -> Vector: ...
    def step_once(self, tick) -> TickOutcome: ...
    def run(self, max_ticks) -> int: ...
    def close(self) -> None: ...
```

`build_host_loop(config, meter=None)` — колонки под фактический `bus_dim`.

## Инварианты

1. `dt > 0`; decay/интегралы в секундах.
2. `synthetic` + детерминированные каналы → replay; ресурсы — исключение.
3. `time_scale` масштабирует субъективное время (1.0 = жизнь).
4. Непрерывность: эмоциональный контур всегда включён (ADR-0006).
5. Non-finite → HostIntegrityError (fail-fast).
6. Телеметрия: 15 плоских полей.

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
- Рациональный контур event-triggered — S3/S4
- Пред-колоночная γ — S4

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Расположение провайдеров | Решено | `src/host/sources.py` |
| Карта сегментов | Решено | заложена сразу |
| Инъекция meter | Решено | `build_host_loop(config, meter)` |
| Частота heartbeat | Решено | 10 Гц (dt=0.1) |
