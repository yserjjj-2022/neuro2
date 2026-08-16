# PLAN.md — src/core/voting

## Файлы для создания/изменения

1. `src/core/voting/models.py` — `VotingResult` (frozen dataclass) + type aliases
2. `src/core/voting/kwta.py` — `kwta()` (Functional Core, чистая функция)
3. `src/core/voting/manager.py` — `VotingManager` (Imperative Shell)
4. `src/core/voting/__init__.py` — re-export (сейчас пустой)
5. `src/core/__init__.py` — добавить re-export voting
6. `src/core/voting/README.md` — обновлён (O(N) = цель Фазы 3)
7. `src/tests/test_voting_kwta.py` — Unit-тесты для Core (`kwta`)
8. `src/tests/test_voting_manager.py` — Unit-тесты для Shell (`VotingManager`)
9. `src/tests/test_integration_voting_cmc.py` — Интеграция cmc → voting

## Зависимости

**Внешние:**
- `numpy` — argsort, mask

**Внутренние:**
- `src/core/cmc` — только в интеграционном тесте (`CMCEnsemble`)

**Стандартная библиотека:**
- `dataclasses` — frozen dataclass
- `typing` — `Any` (для type aliases)

## Порядок реализации

### 1. Models (`models.py`)
- `Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]` — идиома из `src/memory/serialize.py`
- `IndexVector = np.ndarray[Any, np.dtype[np.intp]]` — для индексов (mypy strict требует type args)
- `VotingResult` (frozen): `indices: IndexVector`, `mask: Vector`, `scores: Vector`

### 2. Kwta Core (`kwta.py`)
- `kwta(scores: Vector, k: int) -> VotingResult`
  1. **Validate**: `scores.ndim != 1` → ValueError; `scores.size == 0` → ValueError
  2. **Validate k**: `k < 1` → ValueError; `k > scores.size` → ValueError
  3. `indices = np.argsort(-scores, kind="stable")[:k]` — top-k по убыванию
  4. `mask = np.zeros(scores.size, dtype=np.float64); mask[indices] = 1.0`
  5. `winner_scores = scores[indices].copy()` — копия, не мутированная
  6. Return `VotingResult(indices, mask, winner_scores)`

### 3. Тесты для Core (`src/tests/test_voting_kwta.py`)
- `test_kwta_basic`: top-k корректно выбирает k наибольших
- `test_kwta_topk_property`: min(победители) >= max(проигравшие)
- `test_kwta_k_equals_one`: k=1 → один победитель (hard-WTA)
- `test_kwta_k_equals_n`: k=N → все победители, mask = 1.0
- `test_kwta_ties`: равные scores → меньший индекс побеждает
- `test_kwta_invalid_k_low`: k < 1 → ValueError
- `test_kwta_invalid_k_high`: k > N → ValueError
- `test_kwta_empty_scores`: пустые scores → ValueError
- `test_kwta_2d_scores`: 2D scores → ValueError
- `test_kwta_purity`: вход не мутируется, повторный вызов → тот же результат
- `test_kwta_mask_sum`: mask.sum() == k

### 4. Manager Shell (`manager.py`)
- `VotingManager.__init__(k: int = 1)` — `k < 1` → ValueError
- `self._k: int`, `self._last: VotingResult | None = None`
- `vote(scores: Vector) -> VotingResult` — делегирует `kwta(scores, self._k)`, кэширует в `_last`
- `last` property — `self._last` (None до первого vote)
- `set_k(k: int)` — `k < 1` → ValueError, обновляет `self._k`

### 5. Тесты для Shell (`src/tests/test_voting_manager.py`)
- `test_manager_default_k`: k=1 по умолчанию
- `test_manager_invalid_k_init`: k < 1 → ValueError
- `test_manager_vote`: возвращает VotingResult с корректными индексами
- `test_manager_vote_k_gt_n`: k > N → ValueError
- `test_manager_last_none_before`: last is None до первого vote
- `test_manager_last_after`: last = результат после vote
- `test_manager_set_k`: set_k меняет поведение; k < 1 → ValueError

### 6. Интеграция cmc → voting (`src/tests/test_integration_voting_cmc.py`)
- `CMCEnsemble` с 3 колонками, первый step → активности `‖e‖²` по строкам errors
- `kwta(scores, k=2)` → победители = колонки с наибольшей ошибкой предсказания
- Проверка: `indices` — колонки с максимальными score; `mask.sum() == 2`
- Второй кейс: стабильный вход 200 шагов → ошибки → 0 → активности → 0 → top-k по нулевым scores → победители — меньшие индексы (ties policy)

### 7. Обновление `__init__.py`
- `src/core/voting/__init__.py`: re-export `kwta`, `VotingManager`, `VotingResult`
- `src/core/__init__.py`: добавить voting в импорт и `__all__`

## План тестов

| Тест | Файл | Покрытие | Инвариант |
|------|------|----------|-----------|
| `test_kwta_basic` | kwta | top-k выбор | len(indices) == k |
| `test_kwta_topk_property` | kwta | top-k свойство | min(win) >= max(lose) |
| `test_kwta_k_equals_one` | kwta | k=1 | hard-WTA |
| `test_kwta_k_equals_n` | kwta | k=N | mask = 1.0 |
| `test_kwta_ties` | kwta | равные scores | меньший индекс |
| `test_kwta_invalid_k_low` | kwta | k < 1 | ValueError |
| `test_kwta_invalid_k_high` | kwta | k > N | ValueError |
| `test_kwta_empty_scores` | kwta | пустые scores | ValueError |
| `test_kwta_2d_scores` | kwta | 2D scores | ValueError |
| `test_kwta_purity` | kwta | чистота | вход не мутирован |
| `test_kwta_mask_sum` | kwta | маска | mask.sum() == k |
| `test_manager_default_k` | manager | дефолт | k == 1 |
| `test_manager_invalid_k_init` | manager | k < 1 | ValueError |
| `test_manager_vote` | manager | vote() | VotingResult |
| `test_manager_vote_k_gt_n` | manager | k > N | ValueError |
| `test_manager_last_none_before` | manager | last | None |
| `test_manager_last_after` | manager | last | результат |
| `test_manager_set_k` | manager | set_k | изменение k |
| `test_integration_voting_cmc` | integration | cmc → voting | победители = макс. ошибка |

## Заметки для реализации

- **Stable argsort**: `np.argsort(-scores, kind="stable")[:k]` — ties → меньший индекс (политика из SPEC, инвариант 5)
- **Типизация**: `IndexVector = np.ndarray[Any, np.dtype[np.intp]]` — `np.argsort` возвращает intp. `VotingResult.indices` — именно этот тип (mypy strict: `int64` не подойдёт к `np.intp`?)
  - Проверить: на практике `np.argsort` даёт dtype intp. Если mypy не сводится — использовать `np.ndarray[Any, np.dtype[np.signedinteger[Any]]]` или каст. Решить при реализации.
- **Purity**: `winner_scores = scores[indices].copy()` — обязательно `.copy()`, иначе результат будет view на входной массив
- **mask float64**: 0.0/1.0, не int — совместимость с soft-WTA (Фаза 2)
- **Валидация в vote()**: `k > N` проверяется в `kwta` (по входному scores) — у manager нет N до вызова
- **Debug logging**: fail-fast валидации — `logger.debug` с причиной (best effort, как в energy/cmc)
