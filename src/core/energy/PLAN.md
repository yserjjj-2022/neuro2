# PLAN.md — src/core/energy

## Файлы

1. `src/core/energy/models.py` — `FreeEnergyResult`, `EnergyState` ✅
2. `src/core/energy/calculator.py` — `FreeEnergyCalculator` (Core) ✅
3. `src/core/energy/observer.py` — `EnergyObserver` (Shell) ✅
4. `src/core/energy/precision.py` — `inverse_variance` + `PrecisionEstimator` ✅
5. `src/core/energy/guards.py` — `HostIntegrityError`, `check_finite` ✅
6. `src/core/energy/drift.py` — `DriftDetector` ✅
7. `src/core/energy/__init__.py` — re-exports ✅
8. `src/tests/test_energy_calculator.py` — тесты калькулятора ✅
9. `src/tests/test_energy_observer.py` — тесты observer ✅
10. `src/tests/test_energy_precision.py` — тесты precision ✅
11. `src/tests/test_energy_guards.py` — тесты guards/drift ✅

## Зависимости

**Внешние:** `numpy`. **Стандартная библиотека:** `dataclasses`, `collections.deque`,
`math`, `typing`, `logging`.

## Порядок реализации (S1, выполнено)

1. `EnergyState` + новая сигнатура `compute(error, precision, state, dt)`:
   сглаженная valence, экспоненциальный stress, `dt > 0` fail-fast.
2. `EnergyObserver` владеет `EnergyState`; `observe(..., dt)`.
3. `inverse_variance` (Core) + `PrecisionEstimator` (Shell, окно).
4. `HostIntegrityError`/`check_finite` + `DriftDetector`.
5. `__init__.py`: re-export всех сущностей.

## План тестов

| Тест | Покрытие | Инвариант |
|---|---|---|
| `test_compute_valid` | формула F | F ≥ 0 |
| `test_compute_invalid_dt` | dt ≤ 0 | ValueError |
| `test_valence_smoothing_reduces_jitter` | EMA | сглаживание |
| `test_stress_leak` | утечка | затухание |
| `test_dt_scaling_stress` | f·dt | секунды |
| `test_inverse_variance_*` | γ=1/var | purity, clip, default=10 |
| `test_precision_estimator_*` | окно | bounded, reset |
| `test_check_finite_*` | NaN/inf | HostIntegrityError |
| `test_drift_detector_*` | порог+hold | флаг |
| `test_observer_maintains_state` | EnergyState | перенос |

## Заметки

- **Clip order**: validate dt и shapes до clip (fail-fast).
- **gamma_max=10.0**: калибровка S1; 1e6 взрывал F (см. S1 SPEC).
- **Человеческие константы**: `valence_tau=1.0`, `stress_leak=0.01` — эмоц. темп (ADR-0006).
- **Stateless core**: `compute()` не меняет `self`; состояние — в `EnergyState`.
- **Stateful shell**: `EnergyObserver` владеет `EnergyState`.
