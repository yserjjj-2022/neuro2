# PLAN.md — src/core/attractors

## Файлы для создания/изменения

1. `src/core/attractors/models.py` — `TaskAttraction` (frozen dataclass) + type aliases
2. `src/core/attractors/compute.py` — `compute_dwell`, `check_basin_stability`, `check_immediate_switch` (Core, чистые функции)
3. `src/core/attractors/manager.py` — `TaskAttractor` (Shell)
4. `src/core/attractors/__init__.py` — re-export (сейчас пустой)
5. `src/core/__init__.py` — добавить re-export attractors
6. `src/core/attractors/README.md` — документация
7. `src/tests/test_attractors_compute.py` — Unit-тесты для Core (`compute_dwell`, `check_basin_stability`, `check_immediate_switch`)
8. `src/tests/test_attractors_manager.py` — Unit-тесты для Shell (`TaskAttractor`)
9. `src/tests/test_integration_attractors_cmc.py` — Интеграция cmc → attractors

## Зависимости

**Внешние:**
- `numpy` — argmax, mask operations

**Внутренние:**
- `src/core/cmc` — только в интеграционном тесте (`CMCEnsemble`, `ColumnConfig`)

**Стандартная библиотека:**
- `dataclasses` — frozen dataclass
- `typing` — `Any` (для type aliases)

## Порядок реализации

### 1. Models (`models.py`)
- `Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]` — идиома из `src/core/voting/models.py`
- `TaskAttraction` (frozen): `mask: Vector`, `history_size: int`, `scores: Vector`, `converged: bool`

### 2. Core (`compute.py`)
- `compute_dwell(base_dwell, dwell_slope, plasticity_gain, score_current, score_runner_up, history_size) -> int`
  1. `delta = score_current - score_runner_up`
  2. `raw_dwell = base_dwell + dwell_slope * delta + plasticity_gain * history_size`
  3. `return max(base_dwell, round(raw_dwell))` — жёсткий пол
- `check_basin_stability(basin_threshold, score_current, score_runner_up, history_size) -> bool`
  1. `effective_threshold = basin_threshold + 0.05 * history_size`
  2. `delta = score_current - score_runner_up`
  3. `return delta > -effective_threshold`
- `check_immediate_switch(score_current, score_runner_up, dominance_threshold=0.3) -> bool`
  1. `return (score_runner_up - score_current) > dominance_threshold`

### 3. Тесты для Core (`src/tests/test_attractors_compute.py`)
- `test_compute_dwell_win` — `Δscore > 0` → `min_dwell > base_dwell`
- `test_compute_dwell_lose` — `Δscore < 0` → `min_dwell == base_dwell` (пол держит)
- `test_compute_dwell_equal` — `Δscore == 0` → `min_dwell == base_dwell`
- `test_compute_dwell_stp` — `history_size > 0, gain > 0` → `min_dwell > base_dwell`
- `test_compute_dwell_floor` — `Δscore << 0` → `min_dwell >= base_dwell` (жёсткий пол)
- `test_check_basin_stability_in_basin` — малый Δscore → True
- `test_check_basin_stability_out_of_basin` — большой Δscore → False
- `test_check_basin_stability_stp_boost` — history_size растит effective_threshold
- `test_check_immediate_switch_dominance` — большой Δscore → True
- `test_check_immediate_switch_no_dominance` — малый Δscore → False

### 4. Manager Shell (`manager.py`)
- `TaskAttractor.__init__(n_tasks, base_dwell=5, dwell_slope=2.0, plasticity_gain=0.1, basin_threshold=0.15, convergence_threshold=1e-8)`
  - `n_tasks < 2` → ValueError
  - `self._mask: Vector | None = None`
  - `self._history_size: int = 0`
  - `self._base_dwell: int = base_dwell`
  - `self._dwell_slope: float = dwell_slope`
  - `self._plasticity_gain: float = plasticity_gain`
  - `self._basin_threshold: float = basin_threshold`
  - `self._convergence_threshold: float = convergence_threshold`
- `tick(scores: Vector) -> TaskAttraction`
  0. **Первый тик** (self._mask is None):
     - `mask = np.zeros(n_tasks, dtype=np.float64); mask[np.argmax(scores)] = 1.0`
     - `history_size = 0`
     - `converged = bool(np.max(scores) < self._convergence_threshold)`
     - Return `TaskAttraction(mask, history_size, scores.copy(), converged)`
  1. Найти `held_index` — индекс 1.0 в текущем mask (`np.where(self._mask == 1.0)[0][0]`)
  2. Найти `score_runner_up = max(scores[i] for i in range(n_tasks) if i != held_index)`
  3. Найти `score_current = scores[held_index]`
  4. Если `check_immediate_switch(score_current, score_runner_up)`: switch (mask = one-hot для argmax, history = 0)
  5. Иначе:
     a. Вычислить `min_dwell = compute_dwell(self._base_dwell, self._dwell_slope, self._plasticity_gain, score_current, score_runner_up, self._history_size)`
     b. Если `history_size < min_dwell`: stay (history += 1)
     c. Иначе (history_size >= min_dwell):
        - `switch_now = (np.argmax(scores) != held_index) and not check_basin_stability(self._basin_threshold, score_current, score_runner_up, self._history_size)`
        - Если `switch_now`: switch (mask = one-hot по argmax, history = 0)
        - Иначе (остаёмся — либо held и так топ-1, либо в бассейне): history += 1, stay
  6. `converged = bool(np.max(scores) < self._convergence_threshold)`
  7. Return `TaskAttraction(mask, history_size, scores.copy(), converged)`
- `reset()` — mask = None, history = 0 (следующий tick() попадёт в ветку «первый тик»)
- `current_mask` property — `Vector | None` (None до первого tick())
- `history_size` property

### 5. Тесты для Shell (`src/tests/test_attractors_manager.py`)
- `test_manager_init_n_tasks_lt_2` — `n_tasks < 2` → ValueError
- `test_manager_tick_first` — первый тик: mask = one-hot, history = 0
- `test_manager_tick_stable` — stable input: mask не меняется, history растёт
- `test_manager_tick_switch` — switch: новый mask, history = 0
- `test_manager_tick_convergence` — scores < τ, switch запрещён до history >= min_dwell
- `test_manager_tick_immediate_switch` — явное превосходство конкурента → switch
- `test_manager_tick_basin_stability` — малый Δscore, stay + history_size вырос на 1
- `test_manager_reset` — mask = None, history = 0
- `test_manager_tick_after_reset` — reset() затем tick(scores): mask = one-hot по argmax, history = 0 (как новый объект)
- `test_manager_current_mask_none_before` — None до первого tick()

### 6. Интеграция cmc → attractors (`src/tests/test_integration_attractors_cmc.py`)
- `CMCEnsemble` с 3 колонками, первый step → активности `‖e‖²` по строкам errors
- `TaskAttractor.tick(activities)` → mask = one-hot для колонки с максимальной ошибкой
- Стабильный вход 200 шагов → mask не меняется, history растёт
- Внезапный новый паттерн входа → switch (если конкурент явный)

### 7. Обновление `__init__.py`
- `src/core/attractors/__init__.py`: re-export `TaskAttraction`, `TaskAttractor`, `compute_dwell`, `check_basin_stability`, `check_immediate_switch`
- `src/core/__init__.py`: добавить attractors в импорт и `__all__`

## План тестов

| Тест | Файл | Покрытие | Инвариант |
|------|------|----------|-----------|
| `test_compute_dwell_win` | compute | выигрыш | min_dwell > base_dwell |
| `test_compute_dwell_lose` | compute | проигрыш | min_dwell == base_dwell |
| `test_compute_dwell_equal` | compute | равенство | min_dwell == base_dwell |
| `test_compute_dwell_stp` | compute | STP | min_dwell > base_dwell |
| `test_compute_dwell_floor` | compute | жёсткий пол | min_dwell >= base_dwell |
| `test_check_basin_stability_in_basin` | compute | в бассейне | True |
| `test_check_basin_stability_out_of_basin` | compute | вне бассейна | False |
| `test_check_basin_stability_stp_boost` | compute | STP-усиление | effective_threshold растёт |
| `test_check_immediate_switch_dominance` | compute | доминирование | True |
| `test_check_immediate_switch_no_dominance` | compute | нет доминирования | False |
| `test_manager_init_n_tasks_lt_2` | manager | валидация | ValueError |
| `test_manager_tick_first` | manager | первый тик | mask = one-hot |
| `test_manager_tick_stable` | manager | стабильность | history растёт |
| `test_manager_tick_switch` | manager | переключение | новый mask |
| `test_manager_tick_convergence` | manager | сходимость | switch запрещён |
| `test_manager_tick_immediate_switch` | manager | доминирование | switch |
| `test_manager_tick_basin_stability` | manager | бассейн | stay + history_size вырос на 1 |
| `test_manager_reset` | manager | сброс | mask = None |
| `test_manager_tick_after_reset` | manager | reset+tick | mask = one-hot, history = 0 |
| `test_manager_current_mask_none_before` | manager | None до первого tick() | None |
| `test_integration_attractors_cmc` | integration | cmc → attractors | mask = max error |

## Заметки для реализации

- **Runner-up calculation**: `score_runner_up = max(scores[i] for i in range(len(scores)) if i != held_index)`. Это не просто "топ-2" в отсортированном списке, а лучший результат среди всех колонок, кроме удерживаемой. Технически совпадает с "топ-2", когда held действительно топ-1, но расходится, когда held проигрывает — наивная реализация через `np.argsort(scores)[-2:]` без учёта `held_index` даст неверное значение в этом случае.
- **Dynamic dwell check**: `dwell_remaining` не хранится как отдельное состояние. На каждом тике сравнивается `history_size < compute_dwell(...)`, пересчитывая порог заново из актуальных scores. Это чище с точки зрения FC/IS.
- **Параллельные потребители**: `VotingManager.vote()` и `TaskAttractor.tick()` — параллельные, независимые потребители одного и того же вектора activities (из CMCEnsemble.step()). В wiring.py они вызываются последовательно, но это просто строки кода, а не логическая зависимость данных.
- **Purity**: `compute_dwell`, `check_basin_stability`, `check_immediate_switch` не мутируют входные массивы. Тест purity: изменить входной массив после вызова — результат не меняется. Кроме того, `TaskAttraction.scores` хранит копию `scores.copy()`, а не ссылку на переданный вызывающим кодом массив — иначе, если wiring.py мутирует свой буфер activities между тиками (для переиспользования памяти), исторический снимок в TaskAttraction задним числом изменится, что нарушит purity-гарантию.
- **Type hints**: использовать `Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]` (идиома из `src/core/voting/models.py`). `current_mask` property — тип `Vector | None` (None до первого tick()).
- **Debug logging**: при переключении аттрактора — `logger.debug` с Δscore, min_dwell, reason (basin / convergence / switch) — для калибровки порогов в Фаза 3.
