# PLAN.md — src/core/energy

## Файлы для создания/изменения

1. `src/core/energy/models.py` — `FreeEnergyResult` dataclass
2. `src/core/energy/calculator.py` — `FreeEnergyCalculator` (pure logic)
3. `src/core/energy/observer.py` — `EnergyObserver` (shell + DI)
4. `src/core/energy/__init__.py` — Re-exports (обновить)
5. `src/tests/test_energy_calculator.py` — Unit-тесты для калькулятора
6. `src/tests/test_energy_observer.py` — Unit-тесты для observer

## Зависимости

**Внешние:**
- `numpy` — векторные операции

**Стандартная библиотека:**
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
  1. **Validate shapes** (`ValueError` при несовпадении) — fail-fast
  2. **Clip** `precision` до `1e-6` (с `logger.debug`)
  3. Вычислить `F(t) = 0.5 * sum(gamma * error**2)`
  4. Вычислить `valence`, `stress`, `gamma`
- Вернуть `FreeEnergyResult`

### 3. Тесты для Calculator (`src/tests/test_energy_calculator.py`)
- `test_compute_valid`: Формула F(t) на валидных данных
- `test_compute_shape_mismatch`: ValueError при несовпадении
- `test_compute_empty_arrays`: F(t) = 0.0 для пустых векторов
- `test_compute_empty_precision`: gamma = gamma_base
- `test_compute_precision_clip`: молчаливый clip precision <= 0
- `test_valence_sign`: проверка знака valence
- `test_stress_decay`: монотонное затухание

### 4. Observer (`observer.py`) — Functional Shell

**Ключевой принцип:** Calculator — чистая функция (stateless), Observer — imperative shell, хранит состояние между вызовами.

- Создать `EnergyObserver`
- `__init__`: `calculator`, `sink` (Callable | None)
- **Внутреннее состояние:**
  - `self._prev_f: float = 0.0` — F(t-1)
  - `self._prev_stress: float = 0.0` — stress(t-1)
- `observe()`:
  1. Вызывает `calculator.compute(prediction_error, precision, self._prev_f, self._prev_stress)`
  2. Обновляет `self._prev_f = result.f`, `self._prev_stress = result.allostatic_stress`
  3. Если `sink` задан — вызывает `sink(result)`
  4. Возвращает `result`

**Это единственное состояние в модуле:** живёт в shell, calculator остаётся чистым.

### 5. Тесты для Observer (`src/tests/test_energy_observer.py`)
- `test_observer_no_sink`: observe() возвращает результат без sink
- `test_observer_with_sink`: sink вызывается с результатом
- `test_observer_maintains_state`: два последовательных observe() корректно передают f(t-1)/stress(t-1) во второй вызов

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
| `test_observer_maintains_state` | State между вызовами | Динамика по времени |

## Заметки для реализации

- **Clip order**: validate shapes **до** clip (fail-fast)
- **Debug logging**: `logger.debug("precision clipped")` при клиппинге (не покрыто тестом, best effort)
- **Stateless core**: `compute()` не меняет `self`, состояние — в параметрах
- **Stateful shell**: `Observer` хранит `_prev_f`, `_prev_stress` — это единственное мутабельное состояние
- **Tests**: использовать `numpy.testing` для float comparisons
