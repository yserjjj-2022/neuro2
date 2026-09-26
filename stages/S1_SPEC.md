# SPEC.md — S1: Честные сигналы

Стадия S1 из `BUILD_ROADMAP.md`. Фундамент: без валидных сигналов нельзя
строить ни память, ни речь, ни policy. Цель — превратить рефлекторную дугу
в **читаемый и физически осмысленный** контур.

Ворота S1 — в `VALIDATION.md` §4. Решения — ADR-0005 §2, §7.

## Область (входит)

1. **Единая временная база** — tick ↔ секунды; decay/интегралы в секундах.
2. **EnergyState + валидная valence** — сглаженная производная; `dt` отвязан
   от шага loop.
3. **Настоящая γ** — обратная дисперсия входа (`PrecisionEstimator`).
4. **Расширение телеметрии** — tick, tags, reflex, bus_dim, gamma, resources, drift.
5. **ResourceProvider** — латентность + RSS как интероцептивный сигнал.
6. **Guards** — NaN/inf (fail-fast) + заготовка детектора дрейфа.
7. **Поведенческий регресс** — канонические сценарии + «отпечаток» прогона.

## Явно НЕ входит

- **Пред-колоночная γ** (барьер внимания) — S4.
- **Реакция на перегрузку** (throttle) — S4; в S1 только измерение/логирование.
- **Рефлекс-маршрут** — S4; в S1 только контракт `is_reflex` + логирование.
- **Память/эмбеддер** — S2.
- **Проекции колонок** — S2+; колонки по-прежнему читают всю шину.
- **Действие (policy)** — S4.

## 1. Временная база

### Понятия

- `tick_dt` — номинальная длительность тика в секундах (база интегрирования).
- `heartbeat_hz` — частота эмоционального контура (дефолт 10 Гц, dt = 0.1 с).
- `time_scale` — множитель субъективного времени: `1.0` = жизнь, `>1` = симуляция.
  Режим логируется в телеметрии (защита от «забыли вернуть 1:1»).
- `clock_mode` — источник времени:
  - `"synthetic"` (дефолт): `now = tick · tick_dt`, `dt = tick_dt`.
    Детерминированно, replay-совместимо.
  - `"wall"`: `now = clock()`, `dt = now − prev_now` (первый тик → `tick_dt`).
    Для реального времени; `dt` измеряется, не константа.
- `paced` — спать ли между тиками, чтобы реальное время ≈ `tick_dt`.
  `False` (дефолт) — free-run (тесты, batch).

### Принцип непрерывности (ADR-0006)

- **Эмоциональный контур всегда включён** — это субстрат существования.
  Он не спит между событиями; idle-динамика определена: циркадный дрейф,
  утечка стресса, дрейф F относительно модели, детектор дрейфа.
- **Рациональный контур — event-triggered** поверх субстрата (полный
  пересчёт/LLM/MCP/policy — по условию). В S1 ещё не выделен в коде.

### Инварианты

1. `dt > 0` всегда (loop гарантирует; `tick_dt > 0`).
2. При `clock_mode="synthetic"` прогон детерминирован при фиксированном seed
   (кроме реальных ресурсных каналов — см. §5).
3. Decay и интегралы зависят от `dt` (секунд), а не от номера тика.
4. Эмоциональный контур не имеет разрывов состояния между тиками.

## 2. EnergyState и валидная valence

### EnergyState (frozen dataclass, Functional Core)

```python
@dataclass(frozen=True)
class EnergyState:
    f: float = 0.0
    stress: float = 0.0
    valence: float = 0.0
```

### FreeEnergyCalculator (stateless)

```python
class FreeEnergyCalculator:
    def __init__(
        self,
        stress_leak_per_sec: float = 1.0,   # λ, 1/с
        valence_tau: float = 0.1,           # τ сглаживания valence, с
        gamma_base: float = 1.0,
    ) -> None: ...

    def compute(
        self,
        prediction_error: Vector,
        precision: Vector,
        state: EnergyState,
        dt: float,
    ) -> FreeEnergyResult: ...
```

**Формулы** (при `dt > 0`):

```
f          = 0.5 · Σ γᵢ·eᵢ²                      (пусто → 0.0)
valence_raw = −(f − state.f) / dt
a           = 1 − exp(−dt / valence_tau)
valence     = (1 − a)·state.valence + a·valence_raw      # EMA по времени
stress      = state.stress · exp(−stress_leak_per_sec·dt) + f·dt
gamma       = mean(precision)  (пусто → gamma_base)
```

**Почему так:**
- `valence` сглажена по времени → убирает «дребезг» (649 смен знака).
- `stress` — утечка + накопление `f·dt` (физический интеграл), а не «на тик».
- `dt` передаётся явно → калькулятор не хранит шаг.

### EnergyObserver (Shell)

Хранит `_state: EnergyState`; `observe(error, precision, dt)` обновляет state
из результата. `prev_f`/`prev_stress`/`prev_valence` уходят в `EnergyState`.

### Изменение CMCPipeline.tick

`CMCPipeline.tick(u, precision, dt) -> FreeEnergyResult` (добавлен `dt`).

## 3. Настоящая γ — PrecisionEstimator

### inverse_variance (чистое ядро)

```python
def inverse_variance(samples: Vector, eps: float, gamma_max: float) -> Vector:
    """γᵢ = clip(1 / (varᵢ + eps), 0, gamma_max)."""

class PrecisionEstimator:
    def __init__(self, dim: int, window: int = 50,
                 eps: float = 1e-6, gamma_max: float = 10.0) -> None: ...
    def update(self, u: Vector) -> Vector:
        """Добавить наблюдение u(t), вернуть γ shape=(dim,)."""
```

Окно — скользящее (deque копий `u`, длина `window`). До накопления окна
γ строится по имеющимся сэмплам (первый сэмпл → `gamma_max` при нулевой var).

### Применение в loop

```python
gamma_bus = estimator.update(u)          # (bus_dim,)
precision = np.tile(gamma_bus, n_columns)  # (n_columns·bus_dim,)
```

`precision_mode`: `"variance"` (дефолт S1) или `"ones"` (baseline — фоллбэк FEP,
для сравнения на воротах).

## 4. Расширение TelemetryEvent

```python
@dataclass(frozen=True)
class TelemetryEvent:
    timestamp: float
    tick: int
    free_energy: float
    valence: float
    allostatic_stress: float
    gamma: float
    active_columns: int
    active_tags: str          # "cpu,battery" — каналы с ошибкой > порога
    reflex_tags: str          # "battery" — критические сигналы
    bus_dim: int
    latency_ms: float
    rss_mb: float
    drift: bool               # заготовка детектора дрейфа
    phase: str
    mode: str
```

- Плоская структура сохранена (только примитивы) — инвариант telemetry.
- `TelemetryLogger.log(...)` получает новые параметры с дефолтами
  (обратная совместимость вызовов).
- `active_tags`: теги сегментов шины, где агрегированная по колонкам
  ошибка `‖e‖²` > `active_threshold`. Требует `CMCEnsemble.last_errors`
  и карты сегментов в wiring.
- `reflex_tags`: теги сигналов `is_reflex` из `bus.last_signals`.

## 5. ResourceProvider

### ResourceMeter (Shell)

```python
class ResourceMeter:
    def record_tick(self, latency_s: float) -> None: ...
    @property
    def last_latency_s(self) -> float: ...
    def current_rss_mb(self) -> float: ...   # resource.getrusage().ru_maxrss
```

### ResourceProvider (intero, dim=2)

```python
@dataclass(frozen=True)
class ResourceProvider:
    meter: ResourceMeter
    tick_budget_ms: float = 50.0
    rss_budget_mb: float = 1024.0
    tag: str = "resources"
    category: SignalCategory = INTEROCEPTIVE
    dim: int = 2
    period: int = 1
```

- `data = [latency_norm, rss_norm]`, где `latency_norm = last_latency_ms / tick_budget_ms`.
- `severity = max(latency_norm, rss_norm, 0)`, клип [0,1].
- `severity ≥ 0.9` → `is_reflex=True` (контракт `SignalSource`).
- **Не детерминирован** (реальный мир) — единственное исключение из правила
  чистых провайдеров; в тестах `ResourceMeter` фейкается.

Loop: измеряет `perf_counter` вокруг тика, `meter.record_tick(...)`, пишет
`latency_ms`/`rss_mb` в телеметрию. Ресурсный сигнал влияет на **следующий** тик
(post-hoc), реакция — S4.

## 6. Guards и детектор дрейфа

### Guard конечности (fail-fast)

```python
def check_finite(result: FreeEnergyResult) -> None:
    """Raises HostIntegrityError, если f/valence/stress/gamma не конечны."""
```

`HostLoop` вызывает после `pipeline.tick`; при нарушении — `HostIntegrityError`,
перехватывается в `main` → graceful shutdown (снапшот-заготовка). Не продолжаем
с мусором.

### DriftDetector (заготовка)

```python
class DriftDetector:
    def __init__(self, f_threshold: float, stress_threshold: float,
                 hold_ticks: int = 20) -> None: ...
    def update(self, result: FreeEnergyResult) -> bool:
        """True, если F/stress вышли за границы дольше hold_ticks."""
```

В S1 только вычисляет флаг → в телеметрию (`drift`). Реакция — S6.

## 7. Поведенческий регресс

- `behavioral_fingerprint(events: list[dict]) -> dict` — чистая функция:
  F-профиль (min/max/финал), число reflex, пик stress, смены знака valence,
  доля drift, latency p50/p95.
- Тесты прогоняют C1–C6 (`VALIDATION.md` §3) и проверяют **инварианты
  организма** (§2), а не точные значения.
- Golden-файл — опционально позже; в S1 fingerprint печатается/сравнивается
  по инвариантам.

## Инварианты S1

1. `dt > 0`; decay/интегралы зависят от секунд.
2. `valence` — конечна, сглажена; смен знака на 1000 тиков в осмысленном
   диапазоне (порядка десятков, не сотен).
3. `gamma > 0` всегда; `"ones"` доступен как baseline.
4. Телеметрия: 15 плоских полей, нет NaN/inf.
5. `reflex_tags` непуст при `severity ≥ 0.9`.
6. Ресурсный сигнал измеряется и логируется; при перегрузке `severity` растёт.
7. Non-finite результат → `HostIntegrityError` (не тихое продолжение).
8. `clock_mode="synthetic"` → детерминированный прогон.

## Критерии приёмки

См. `VALIDATION.md` §4 (ворота S1). Кратко:
- [ ] C1: F→0, без NaN
- [ ] C2: скачок → F↑, valence<0, затем сходимость
- [ ] valence: смены знака в осмысленном диапазоне
- [ ] C3: reflex виден в телеметрии (`reflex_tags`)
- [ ] C6: ресурсный сигнал в телеметрии
- [ ] `precision_mode="variance"` работает; `"ones"` — baseline
- [ ] non-finite → `HostIntegrityError`
- [ ] `ruff check`/`ruff format` чисто; все тесты зелёные

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Дефолт `clock_mode` | Решено | `"synthetic"` — детерминизм и replay |
| Частота эмоционального контура | Решено | 10 Гц (dt=0.1 с), калибруется (ADR-0006) |
| Разделение временных шкал | Решено | ADR-0006: эмоц./рацион./симуляция |
| Непрерывность | Решено | Эмоциональный контур всегда включён (ADR-0006) |
| `stress_leak_per_sec` дефолт | Решено | 0.01/с (half-life ≈ 69 с, человеческое настроение) |
| `valence_tau` дефолт | Решено | 1.0 с (эмоция); калибровка на воротах |
| `gamma_max` дефолт | Решено | 10.0 (не 1e6 — иначе взрыв F/стресса) |
| `precision_window` | Открыто | 50 тиков; калибровка по C6 |
| Формат `active_tags` | Решено | CSV-строка (плоский JSON) |
| Golden-файл регресса | Отложено | Инварианты в S1; точный эталон — позже |
| `time_scale` в S1 | Решено | Параметр есть, дефолт 1.0; множитель частот — калибруется |
