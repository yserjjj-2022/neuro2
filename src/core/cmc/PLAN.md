# PLAN.md — src/core/cmc

## Файлы для создания/изменения

1. `src/core/cmc/models.py` — `ColumnConfig`, `ColumnState`, `EnsembleOutput` (frozen dataclass)
2. `src/core/cmc/column.py` — `column_step()` (Functional Core, чистая функция)
3. `src/core/cmc/ensemble.py` — `CMCEnsemble` (Imperative Shell)
4. `src/core/cmc/__init__.py` — переписать: удалить `Column`, re-export новых сущностей
5. `src/core/__init__.py` — обновить: `from .cmc import Column` → re-export новых сущностей, исправить docstring
6. `src/core/cmc/README.md` — обновить описание (убрать «SIMD-оптимизация», однослойный Column)
7. `src/tests/test_cmc_column.py` — Unit-тесты для Core (`column_step`, `ColumnConfig`)
8. `src/tests/test_cmc_ensemble.py` — Unit-тесты для Shell (`CMCEnsemble`)
9. `src/tests/test_integration_cmc_energy.py` — Интеграция cmc → energy

## Зависимости

**Внешние:**
- `numpy` — векторные операции

**Внутренние:**
- `src/core/energy` — только в интеграционном тесте (`FreeEnergyCalculator`)

**Стандартная библиотека:**
- `dataclasses` — frozen dataclass
- `typing` — `Any` (для `Vector` type alias)
- `logging` — debug-логи fail-fast валидаций (Фаза 2)

## Порядок реализации

### 1. Models (`models.py`)
- `Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]` — идиома из `src/memory/serialize.py`
- `ColumnConfig` (frozen): `input_dim`, `state_dim`, `specialization="general"`, `alpha=0.1`
  - `__post_init__`: валидация `input_dim > 0`, `state_dim > 0`, `state_dim == input_dim`, `0 <= alpha <= 1` → `ValueError`
- `ColumnState` (frozen): `x: Vector`, `e: Vector`
- `EnsembleOutput` (frozen): `errors: np.ndarray`, `states: np.ndarray`, `active: int`

### 2. Column Core (`column.py`)
- `column_step(cfg: ColumnConfig, u: np.ndarray, prev: ColumnState) -> ColumnState`
  1. **Validate shape**: `u.shape != prev.x.shape` → `ValueError` (fail-fast)
  2. `x_new = prev.x + cfg.alpha * (u - prev.x)`
  3. `e_new = u - x_new`
  4. Return `ColumnState(x=x_new, e=e_new)` — **новые массивы**, не ссылки на входы

### 3. Тесты для Core (`src/tests/test_cmc_column.py`)
- `test_config_invalid_dims`: `input_dim <= 0` → ValueError
- `test_config_invalid_alpha`: `alpha < 0` и `alpha > 1` → ValueError
- `test_config_state_neq_input`: `state_dim != input_dim` → ValueError
- `test_column_step_first`: из нулевого состояния → `e(0) = u(0)`
- `test_column_step_purity`: повторный вызов с теми же аргументами → тот же результат; мутация `u` после вызова не влияет
- `test_column_step_convergence`: стабильный вход, N шагов → `||e|| < tolerance`
- `test_column_step_shape_mismatch`: `u.shape != prev.x.shape` → ValueError

### 4. Ensemble Shell (`ensemble.py`)
- `CMCEnsemble.__init__(columns: list[ColumnConfig], active_threshold: float = 1e-8)`
  1. Валидация: пустой список → ValueError; разные `input_dim`/`state_dim` → ValueError
  2. Сохранить конфиги, инициализировать состояния нулями: `self._states: list[ColumnState]`
- `step(u: np.ndarray) -> EnsembleOutput`
  1. Validate `u.shape == (input_dim,)` → ValueError
  2. Для каждой колонки: `column_step(cfg, u, prev)` → обновить `self._states[i]`
  3. Агрегировать: `errors = np.stack([s.e for s in states])`, `states = np.stack([s.x ...])`
  4. `active = sum(1 for s in states if np.sum(s.e ** 2) > active_threshold)`
  5. Return `EnsembleOutput(errors, states, active)`
- `reset()`: все `self._states` → `ColumnState(zeros, zeros)`
- `active` property: `self._active` (кэш после step), до первого step — 0

### 5. Тесты для Shell (`src/tests/test_cmc_ensemble.py`)
- `test_ensemble_init_empty`: пустой список колонок → ValueError
- `test_ensemble_init_mismatched_dims`: разные input_dim → ValueError
- `test_ensemble_step_aggregation`: N колонок → errors shape `(N, input_dim)`, states shape `(N, state_dim)`
- `test_ensemble_step_shape_mismatch`: `u.shape != (input_dim,)` → ValueError
- `test_ensemble_active`: колонки с нулевой ошибкой → active=0; с ненулевой → active считает правильно
- `test_ensemble_reset`: после reset все состояния нулевые, active == 0
- `test_ensemble_active_before_step`: `active` property == 0 до первого step

### 6. Интеграция cmc → energy (`src/tests/test_integration_cmc_energy.py`)
- Создать `CMCEnsemble` с 3 колонками, `FreeEnergyCalculator`
- Подать стабильный вход N раз через `ensemble.step(u)`
- На каждом шаге: `errors = output.errors.ravel()`, `precision = np.ones_like(errors)`
- `calc.compute(errors, precision, prev_f, prev_stress)` → `FreeEnergyResult`
- Утверждение: `F(t)` убывает к 0 по мере сходимости `e(t) → 0`
- Дополнительно: `output.active` передаётся как `active_columns` (паттерн wiring)

### 7. Обновление `__init__.py` файлов
- `src/core/cmc/__init__.py`: удалить `Column`, re-export `ColumnConfig`, `ColumnState`, `column_step`, `CMCEnsemble`, `EnsembleOutput`
- `src/core/__init__.py`: `from .cmc import Column` → re-export новых сущностей, обновить docstring (убрать «SIMD-оптимизация», описать FC/IS)
- `src/core/cmc/README.md`: обновить описание

## План тестов

| Тест | Файл | Покрытие | Инвариант |
|------|------|----------|-----------|
| `test_config_invalid_dims` | column | input_dim <= 0 | ValueError fail-fast |
| `test_config_invalid_alpha` | column | alpha ∉ [0,1] | ValueError fail-fast |
| `test_config_state_neq_input` | column | state_dim != input_dim | ValueError fail-fast |
| `test_column_step_first` | column | первый шаг из нулей | e(0) = u(0) |
| `test_column_step_purity` | column | чистота функции | одинаковый вход → выход, no mutation |
| `test_column_step_convergence` | column | стабильный вход | e → 0 |
| `test_column_step_shape_mismatch` | column | u.shape != x.shape | ValueError |
| `test_ensemble_init_empty` | ensemble | пустой список | ValueError |
| `test_ensemble_init_mismatched_dims` | ensemble | разные размерности | ValueError |
| `test_ensemble_step_aggregation` | ensemble | N колонок | errors/states shape |
| `test_ensemble_step_shape_mismatch` | ensemble | u.shape != input_dim | ValueError |
| `test_ensemble_active` | ensemble | подсчёт активных | ‖e‖² > threshold |
| `test_ensemble_reset` | ensemble | сброс | x=0, e=0, active=0 |
| `test_ensemble_active_before_step` | ensemble | active до step | 0 |
| `test_integration_cmc_energy` | integration | cmc → energy | F(t) → 0 при сходимости |

## Заметки для реализации

- **Vector type alias**: `np.ndarray[Any, np.dtype[np.floating[Any]]]` — numpy-дженерики инвариантны, `np.floating[Any]` принимает и float32, и float64 (идиома из `src/memory/serialize.py`)
- **Чистота column_step**: `x_new` и `e_new` — результаты арифметических операций (`prev.x + ...`, `u - ...`), NumPy создаёт новые массивы. Тест purity: изменить `u` после вызова — результат не меняется
- **Единые размерности**: все колонки ансамбля имеют одинаковые `input_dim`/`state_dim` — валидация в `__init__`, не в `step`
- **active_threshold дефолт 1e-8**: EMA never converges to exact 0.0 in float64 (свойство рекуррентного фильтра, не баг). Значение 1e-8 предотвращает ложноположительные active_columns при длительной сходимости. `np.sum(e ** 2) > 1e-8` — корректное сравнение.
- **Stacking**: `np.stack([s.e for s in states])` → shape `(N, input_dim)`. Не `np.array` (может дать object array при разных shapes — но валидация гарантирует одинаковые)
- **Debug logging**: `logger.debug` при fail-fast валидациях — best effort, не покрывается тестом
- **Удаление Column**: `src/core/__init__.py` — единственный импортёр `Column` (подтверждено grep). Одновременное обновление `__init__.py` + `cmc/__init__.py` гарантирует, что импорт пакета не сломается
