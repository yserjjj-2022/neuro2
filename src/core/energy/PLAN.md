# PLAN.md — src/core/energy

## Файлы для создания/изменения

1. `src/core/energy/models.py` — `FreeEnergyResult` dataclass
2. `src/core/energy/calculator.py` — `FreeEnergyCalculator` (pure logic)
3. `src/core/energy/observer.py` — `EnergyObserver` (shell + DI)
4. `src/core/energy/__init__.py` — Re-exports (обновить)
5. `tests/test_energy_calculator.py` — Unit-тесты для калькулятора
6. `tests/test_energy_observer.py` — Unit-тесты для observer

## Зависимости

- `numpy` — векторные операции
- `dataclasses` — frozen dataclass для результата
- `typing` — Callable, Optional
- `logging` — для debug-логов клиппинга (фаза 2)

## Порядок реализации

### 1. Models (`models.py`)
- Создать `@dataclass(frozen=True) FreeEnergyResult`
- Поля: `f`, `valence`, `allostatic_stress`, `gamma` (без `would_trigger`)

### 2. Calculator (`calculator.py`)
- Создать `FreeEnergyCalculator`
- `__init__`: `dt`, `stress_decay`, `gamma_base`
- `compute()`:
  1. Clip `precision` до `1e-6` (с `logger.debug`)
  2. Validate shapes (`ValueError`)
  3. Вычислить `F(t) = 0.5 * sum(gamma * error**2)`
  4. Вычислить `valence`, `stress`, `gamma`
- Вернуть `FreeEnergyResult`

### 3. Тесты для Calculator (`tests/test_energy_calculator.py`)
- `test_compute_valid`: Формула F(t) на валидных данных
- `test_compute_shape_mismatch`: ValueError при несовпадении
- `test_compute_empty_arrays`: F(t) = 0.0 для пустых векторов
- `test_compute_empty_precision`: gamma = gamma_base
- `test_compute_precision_clip`: молчаливый clip precision <= 0
- `test_valence_sign`: проверка знака valence
- `test_stress_decay`: монотонное затухание

### 4. Observer (`observer.py`)
- Создать `EnergyObserver`
- `__init__`: `calculator`, `sink` (Callable | None)
- `observe()`: вызывает `calculator.compute()`, пишет в `sink`

### 5. Тесты для Observer (`tests/test_energy_observer.py`)
- `test_observer_no_sink`: observe() возвращает результат без sink
- `test_observer_with_sink`: sink вызывается с результатом

### 6. `__init__.py`
- Re-export: `FreeEnergyResult`, `FreeEnergyCalculator`, `EnergyObserver`

## План тестов

| Тест | Покрытие | Инвариант |
|------|----------|-----------|
| `test_compute_valid` | Формула F(t) | F(t) ≥ 0 |
| `test_compute_shape_mismatch` | Shape validation | ValueError |
| `test_compute_empty_arrays` | Edge case: пустые векторы | F(t) = 0.0 |
| `test_compute_empty_precision` | Edge case: пустой precision | gamma = gamma_base |
| `test_compute_precision_clip` | Clip precision | γ > 0 |
| `test_valence_sign` | Валентность | Valence = -dF/dt |
| `test_stress_decay` | Стресс | Монотонное затухание |
| `test_observer_no_sink` | Observer без sink | Не блокирует |
| `test_observer_with_sink` | Observer с sink | DI работает |

## Заметки для реализации

- **Clip order**: `precision` клиппится **до** расчёта F(t) и gamma
- **Debug logging**: `logger.debug("precision clipped")` при клиппинге
- **Stateless**: `compute()` не меняет `self`, состояние — в параметрах
- **Tests**: использовать `numpy.testing` для float comparisons
