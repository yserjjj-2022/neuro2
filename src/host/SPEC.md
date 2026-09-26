# SPEC.md — src/host

## Назначение

Host-слой связывает независимо разработанные модули в работающую систему и
оживляет её во времени. Два компонента:

- **sources.py** — сенсорная шина: провайдеры сигналов (mock/real) → `u(t)`.
- **loop.py** — host loop: цикл по `dt`, сборка `u(t)` и `precision`, вызов
  `CMCPipeline.tick()`.
- **wiring.py** — сборка per-tick конвейера (cmc → voting/attractors → energy
  → telemetry).

См. также:
- `src/mcp/SPEC.md` — контракт `SignalSource`/`SignalRegistry` (агрегация)
- `src/config/SPEC.md` — параметры (CONSTITUTION §2.2)
- Манифест §3.Ж (MCP как органы чувств), §3.Б (непрерывное время)

## Поток одного тика

```
now  = clock()
u(t) = SignalBus.step(tick, now)        # шина: конкатенация сигналов
γ    = ones(N_columns * bus_dim)        # precision baseline (Фаза 1)
CMCPipeline.tick(u, γ)                  # cmc → voting/attractors → energy
                                        # → telemetry (JSONL)
```

## Провайдеры сигналов (sources.py)

### SignalProvider (Protocol)

```python
class SignalProvider(Protocol):
    tag: str
    category: SignalCategory
    dim: int
    period: int

    def read(self, tick: int, now: float) -> SignalSource: ...
```

Детерминизм: одинаковые `(tick, now)` → одинаковый `SignalSource`.
Wall-clock владеет loop и передаёт `now` в провайдер.

### Реализации

| Провайдер | Категория | dim | Поведение |
|---|---|---|---|
| `CircadianProvider` | extero | 2 | `[sin, cos]` фазы суток из `now` |
| `BatteryProvider` | intero | 1 | линейный разряд, `severity = 1 - level` |
| `CpuProvider` | intero | 1 | seeded random walk, `severity = load` |
| `UserMessageProvider` | communicative | embedding_dim | фиксированный вектор-заглушка |
| `ConstantProvider` | extero | len(value) | константа (тест сходимости) |
| `StepProvider` | extero | len(before) | скачок на `step_at` (тест реакции) |
| `NoisyProvider` | extero | dim | seeded белый шум (тест стресса) |

`default_providers()` = circadian(2) + battery(1) + cpu(1) + user_message(8)
→ ширина шины **12**.

### SignalBus (Imperative Shell)

```python
class SignalBus:
    def __init__(self, providers: list[SignalProvider]) -> None: ...
    def step(self, tick: int, now: float) -> Vector: ...
    @property
    def segments(self) -> tuple[BusSegment, ...]: ...  # (name, offset, dim, period)
    @property
    def bus_dim(self) -> int: ...
    @property
    def last_signals(self) -> list[SignalSource]: ...
```

Инварианты:
1. **Теги уникальны**, `dim > 0`, `period >= 1` → иначе `ValueError`.
2. **Шина = конкатенация** сигналов (делегируется `SignalRegistry.aggregate()`).
3. **Медленный такт**: `period > 1` → сигнал читается на тиках `tick % period == 0`,
   между ними кэшируется (манифест §3.Ж).
4. **Карта сегментов** — фундамент width scaling: колонка сможет читать свой
   срез шины (`ColumnConfig.reads`, Фаза 2).

## HostLoop (loop.py)

```python
@dataclass
class HostLoop:
    bus: SignalBus
    pipeline: CMCPipeline
    dt: float = 0.01
    precision_mode: str = "ones"
    clock: Callable[[], float] = time.time

    def precision(self) -> Vector: ...
    def run(self, max_ticks: int) -> int: ...
    def step_once(self, tick: int) -> FreeEnergyResult: ...
    def close(self) -> None: ...
```

`build_host_loop(config: HostConfig | None) -> HostLoop` — колонки создаются
под фактический `bus_dim` (конфигурация не разъезжается).

Инварианты:
1. `dt < 0` → `ValueError`; `dt = 0` → без пауз (тесты, batch).
2. `precision_mode ∉ {ones, variance}` → `ValueError`.
3. `precision_mode = "variance"` → `NotImplementedError` (задел Фазы 2).
4. `max_ticks < 0` → `ValueError`.
5. `close()` идемпотентен (graceful shutdown).

## CLI (`python -m src`)

```
uv run python -m src --ticks 100 --dt 0.01 --log run.jsonl \
    [--k 2] [--seed 0] [--message-dim 8] [--precision ones]
```

`--ticks 0` → бесконечный цикл до Ctrl+C (SIGINT → корректное завершение
после текущего тика). Аргументы → `HostConfig` → `build_host_loop`.

## Критерии приёмки

- [x] Провайдеры детерминированы (одинаковый вход → одинаковый выход)
- [x] `severity ≥ 0.9` у interoceptive → `is_reflex=True` (контракт `SignalSource`)
- [x] Шина конкатенирует сигналы, `bus_dim` = сумма dim
- [x] Карта сегментов непрерывна и покрывает всю шину
- [x] Медленный такт: `period > 1` кэширует сигнал
- [x] `HostLoop` гонит N тиков → N событий в JSONL
- [x] Эксперименты #1–#5 (сходимость, скачок, reflex, стресс, дефолт)
- [x] `ruff check` проходит; 222 теста проходят

## Явно НЕ входит в скоуп

- **Реальные интеграции** (datetime/psutil/MCP): провайдеры взаимозаменяемы
  без смены интерфейса (Фаза 2, BACKLOG)
- **Проекции колонок**: Фаза 1 — колонка читает всю шину
- **Оконная дисперсия precision**: задел `precision_mode="variance"` — Фаза 2
- **Reflex routing**: контракт `is_reflex` есть, маршрут — Фаза 3
- **Расширение TelemetryEvent** (`tick`, `active_tags`, `bus_dim`): Фаза 2
- **Память в loop**: нет эмбеддера (BACKLOG Blocked)

## Open Questions

| Вопрос | Статус | Решение |
|--------|--------|---------|
| Расположение провайдеров | Решено | `src/host/sources.py` |
| Карта сегментов | Решено | Заложена сразу — фундамент width scaling |
| precision baseline | Решено | `ones` (Фаза 1), `variance` — Фаза 2 |
| Медленный такт | Решено | Поле `period` у провайдера |
