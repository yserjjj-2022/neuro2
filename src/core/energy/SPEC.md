# SPEC.md — src/core/energy

## Назначение

Аффективный контур хоста: свободная энергия F(t), валентность, аллостатический
стресс, точность γ, а также метакогнитивные guard'ы (контроль целостности,
детектор дрейфа). Модуль экспортирует наблюдаемые метрики, не принимая решений.

Стадия S1 (`stages/S1_SPEC.md`): единая временная база (секунды), сглаженная
valence, настоящая γ (обратная дисперсия). Решения — ADR-0006.

См. также:
- ADR-0004 — FC/IS (`calculator.py` — Core, `observer.py` — Shell)
- ADR-0006 — перманентное существование и временные шкалы
- `src/telemetry/SPEC.md` — потребитель метрик
- `src/host/SPEC.md` — где вызывается (loop)

## Формулы (S1)

```
F(t)         = 0.5 · Σᵢ γᵢ·e(t)ᵢ²                    (пусто → 0.0)
valence_raw  = -(F(t) - state.f) / dt
a            = 1 - exp(-dt / valence_tau)
valence      = (1 - a)·state.valence + a·valence_raw   # EMA по времени
stress       = state.stress · exp(-λ·dt) + F(t)·dt      # утечка + интеграл
gamma        = mean(precision)  (пусто → gamma_base)
```

где:
- `dt` — шаг интегрирования в секундах (> 0), передаётся явно;
- `valence_tau` (τ) — постоянная времени сглаживания valence, с;
- `stress_leak_per_sec` (λ) — скорость утечки стресса, 1/с;
- `γ` — precision (доверие каналу), из `PrecisionEstimator` или baseline.

Все временные величины — в секундах (единая база, ADR-0006).

## Публичный интерфейс

### FreeEnergyResult (frozen dataclass)

```python
@dataclass(frozen=True)
class FreeEnergyResult:
    f: float  # F(t) ≥ 0
    valence: float  # -dF/dt (сглаженная)
    allostatic_stress: float  # интеграл F(t) с утечкой
    gamma: float  # агрегат precision
```

### EnergyState (frozen dataclass)

```python
@dataclass(frozen=True)
class EnergyState:
    f: float = 0.0  # F(t-1)
    stress: float = 0.0  # stress(t-1)
    valence: float = 0.0  # valence(t-1), для EMA
```

### FreeEnergyCalculator (Core, stateless)

```python
class FreeEnergyCalculator:
    def __init__(
        self,
        stress_leak_per_sec: float = 1.0,
        valence_tau: float = 0.1,
        gamma_base: float = 1.0,
    ) -> None: ...

    def compute(
        self,
        prediction_error: np.ndarray,
        precision: np.ndarray,
        state: EnergyState,
        dt: float,
    ) -> FreeEnergyResult: ...
```

Raises: `ValueError` при `shape mismatch` или `dt <= 0`.

### EnergyObserver (Shell)

```python
class EnergyObserver:
    def __init__(
        self,
        calculator: FreeEnergyCalculator,
        sink: Callable[[FreeEnergyResult], None] | None = None,
    ) -> None: ...

    def observe(self, prediction_error, precision, dt) -> FreeEnergyResult: ...

    @property
    def state(self) -> EnergyState: ...
```

Владеет `EnergyState`; sink — DI (в проде loop логирует сам, observer.sink=None).

### PrecisionEstimator (Shell) + inverse_variance (Core)

```python
def inverse_variance(samples, eps=1e-6, gamma_max=10.0) -> Vector:
    """γ = clip(1/(var+eps), 0, gamma_max)."""


class PrecisionEstimator:
    def __init__(self, dim, window=50, eps=1e-6, gamma_max=10.0) -> None: ...
    def update(self, u: Vector) -> Vector: ...
    @property
    def count(self) -> int: ...
    def reset(self) -> None: ...
```

`gamma_max=10.0` — «во сколько раз максимум доверяем каналу». 1e6 вызывал
взрыв F/stress (см. S1 SPEC).

### Guards

```python
class HostIntegrityError(RuntimeError): ...


def check_finite(result: FreeEnergyResult) -> None:
    """Raises HostIntegrityError при NaN/inf в f/valence/stress/gamma."""
```

### DriftDetector (заготовка S1)

```python
class DriftDetector:
    def __init__(self, f_threshold, stress_threshold, hold_ticks=20) -> None: ...
    def update(self, result: FreeEnergyResult) -> bool: ...
    @property
    def streak(self) -> int: ...
    def reset(self) -> None: ...
```

## Инварианты

1. **F(t) ≥ 0** (квадратичная форма).
2. **valence** — сглажена по времени; на стабильном входе не дребезжит.
3. **stress** — утечка + интеграл `F·dt` (зависит от секунд, не тиков).
4. **γ > 0** всегда; клип до 1e-6 при precision ≤ 0.
5. **dt > 0** — fail-fast.
6. **Non-finite → HostIntegrityError** (не тихое продолжение).
7. **Fully stateless calculator**: всё состояние — в `EnergyState`.
8. **Non-blocking**: compute() быстро на батче ≤ 1000 колонок.

## Критерии приёмки (S1)

- [x] `compute()` — stateless, `dt` явный, `EnergyState` явный
- [x] valence сглажена (`a = 1-exp(-dt/τ)`)
- [x] stress — экспоненциальная утечка + `f·dt`
- [x] `dt <= 0` → ValueError
- [x] `inverse_variance` — чистая, `gamma_max` клип
- [x] `PrecisionEstimator` — окно наблюдений
- [x] `check_finite` → HostIntegrityError
- [x] `DriftDetector` — порог + hold
- [x] 286 тестов, ruff чист

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| `stress_leak_per_sec` | Решено | 0.01/с (настроение, минуты) |
| `valence_tau` | Решено | 1.0 с (эмоция) |
| `gamma_max` | Решено | 10.0 |
| `gamma_base` | Решено | 1.0 |
| Пороги дрейфа | Решено | 100 / 50 (в HostConfig) |
| γ пред-колоночно | Отложено | S4 (ADR-0005 §2) |
