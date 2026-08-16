# SPEC.md — src/core/cmc

## Назначение

Канонические колоночные микроконтуры (CMC): генерация предсказаний по узким проекциям реальности (специализации) и ошибок предсказания e(t) для downstream-модулей (energy → F(t), telemetry → active_columns, voting → k-WTA). Модуль считает и агрегирует — не принимает решений.

Слои колонки: L4 (вход u(t)) → L5/6 (состояние x(t)) → L2/3 (ошибка предсказания e(t)).

Фаза 1 — простая непрерывная динамика на NumPy: локальные обновления per-column (без backprop, без обмена между колонками — латеральные связи это модуль `voting`).

См. также:
- ADR-0004 — Functional Core / Imperative Shell для всех `src/core/*` (column.py = Core, ensemble.py = Shell)
- `src/core/energy/SPEC.md` — потребитель e(t); `FreeEnergyCalculator.compute(prediction_error=..., precision=...)`
- BACKLOG `[Phase3][cmc] Task 1` — прокинуть `active_columns` из cmc в `telemetry_logger.log()`

## Публичный интерфейс

### Формула динамики колонки (Фаза 1)

```
x(t) = x(t-1) + α · (u(t) − x(t-1))
e(t) = u(t) − x(t)
```

где:
- `x(t)` — состояние L5/6 (экспоненциальное скользящее среднее входа)
- `u(t)` — вход L4 (вектор восприятия колонки)
- `e(t)` — ошибка предсказания L2/3 (для energy)
- `α` — скорость обновления состояния, `α ∈ [0, 1]` (параметризуемая)

Свойства:
- При стабильном входе `u(t) = const`: `x(t) → u`, `e(t) → 0` (геометрическая сходимость)
- `α = 0`: состояние не обновляется, `e(t) = u(t) − x(0)`
- `α = 1`: мгновенное копирование входа, `e(t) = 0`
- Начальное состояние: `x(0) = 0`, `e(0) = u(0)`

**Пограничные случаи:**
- `u.shape != x.shape` → `ValueError` (fail-fast, до вычислений)
- `α < 0` или `α > 1` → `ValueError` при создании конфига (fail-fast, не клиппинг)
- `input_dim <= 0` или `state_dim <= 0` → `ValueError` при создании конфига

### ColumnConfig (frozen dataclass, Functional Core)

```python
@dataclass(frozen=True)
class ColumnConfig:
    """Конфигурация одной колонки. Только параметры — состояние не здесь."""

    input_dim: int           # Размерность входа L4
    state_dim: int           # Размерность состояния L5/6
    specialization: str = "general"  # Тег: "tone", "rhythm", "meaning", "tom", "mcp", ...
    alpha: float = 0.1       # Скорость обновления состояния, α ∈ [0, 1]
```

Note:
- `specialization` в Фазе 1 — тег, не влияет на вычисления. Проекции входа по специализациям — Фаза 2 (см. Open Questions).
- `state_dim == input_dim` в Фазе 1 (матрицы проекций нет). Несовпадение → `ValueError` в конфиге.
- Дефолты только для `alpha` и `specialization`; размерности — обязательные аргументы (без неявных значений).

### ColumnState (frozen dataclass, Functional Core)

```python
@dataclass(frozen=True)
class ColumnState:
    """Иммутабельный снимок состояния колонки: x(t) и e(t)."""

    x: np.ndarray            # Состояние L5/6, shape == (state_dim,)
    e: np.ndarray            # Ошибка предсказания L2/3, shape == (state_dim,)
```

Note:
- Не мутируется `column_step` — функция чистая, `prev` передаётся явно.
- Аналогия: `Episode` (memory) и `FreeEnergyResult` (energy) — frozen dataclass как снимок.

### column_step (чистая функция, Functional Core)

```python
def column_step(cfg: ColumnConfig, u: np.ndarray, prev: ColumnState) -> ColumnState:
    """Один шаг динамики колонки.

    Чистая функция: не мутирует prev и u, не хранит состояние.
    Состояние передаётся явно — вызывающий код (ensemble) владеет им.

    Args:
        cfg: Конфигурация колонки.
        u: Вход L4, shape == (input_dim,).
        prev: Состояние на предыдущем шаге x(t-1), e(t-1).

    Returns:
        Новый ColumnState: x(t), e(t).

    Raises:
        ValueError: Если u.shape != prev.x.shape (или != cfg.input_dim).

    Formula:
        x(t) = x(t-1) + alpha · (u(t) − x(t-1))
        e(t) = u(t) − x(t)
    """
    ...
```

Note:
- Первый шаг: `prev = ColumnState(x=zeros(input_dim), e=zeros(input_dim))` — инициализация вызывающим кодом (ensemble), не внутри функции.
- Функция возвращает новый объект; контракт: входные массивы не изменяются.

### EnsembleOutput (frozen dataclass, Imperative Shell)

```python
@dataclass(frozen=True)
class EnsembleOutput:
    """Снимок всего ансамбля после step()."""

    errors: np.ndarray       # e(t) всех колонок, shape == (N, input_dim)
    states: np.ndarray       # x(t) всех колонок, shape == (N, state_dim)
    active: int              # Число активных колонок (для telemetry active_columns)
```

### CMCEnsemble (Imperative Shell)

```python
class CMCEnsemble:
    """Imperative Shell: владеет состояниями всех колонок.

    Functional Core / Imperative Shell (ADR-0004):
    - Core (column_step) — чистая функция, тестируется без Shell
    - Shell (CMCEnsemble) — хранит состояния, агрегирует, считает active
    - В проде: step(u) вызывается каждый тик хоста, результаты передаются
      в energy (e(t)) и telemetry (active)
    """

    def __init__(
        self,
        columns: list[ColumnConfig],
        active_threshold: float = 0.0,
    ) -> None:
        """Создание ансамбля: валидация, инициализация состояний нулями.

        Args:
            columns: Конфигурации колонок. Все должны иметь одинаковые
                input_dim и state_dim (единый тензор [N, In, State],
                конституция §2.1).
            active_threshold: Колонка активна, если ||e(t)||² > threshold.

        Raises:
            ValueError: Если список пуст, или input_dim/state_dim
                различаются между колонками, или размерности <= 0.
        """
        ...

    def step(self, u: np.ndarray) -> EnsembleOutput:
        """Один тик: прогнать все колонки через column_step, агрегировать.

        Args:
            u: Вход ансамбля L4, shape == (input_dim,). Один вектор,
                общий для всех колонок (каждая берёт свою проекцию —
                в Фазе 1 проекция = весь вектор, см. Open Questions).

        Returns:
            EnsembleOutput: errors, states, active.

        Raises:
            ValueError: Если u.shape != (input_dim,).
        """
        ...

    def reset(self) -> None:
        """Сбросить все состояния колонок в нули (x = 0, e = 0)."""
        ...

    @property
    def active(self) -> int:
        """Число активных колонок после последнего step().

        Вычисляется как: count(col for col in columns if ||e_col||² > threshold).
        До первого step() — 0.
        """
        ...
```

## Инварианты

1. **FC/IS**: `column_step` — чистая функция (одинаковый `(cfg, u, prev)` → одинаковый результат; входные массивы не мутируются). `CMCEnsemble` — единственный владелец состояний.
2. **Сходимость**: при стабильном `u` ошибка `e(t) → 0` (колонка «привыкает» к входу).
3. **Локальные обновления**: состояние колонки зависит только от её собственного входа и состояния (конституция §2.1 — никакого обмена между колонками, никакого backprop через ансамбль).
4. **Единые размерности**: все колонки ансамбля имеют одинаковые `input_dim`/`state_dim` — единый тензор `[N, In, State]`.
5. **Fail-fast**: любые несоответствия размерностей и `α ∉ [0, 1]` → `ValueError` до вычислений (не клиппинг, не молчаливая коррекция).
6. **Параметризуемость**: `alpha`, `active_threshold` — параметры конструкторов, не хардкод (конституция §2.2).
7. **Non-blocking**: `step()` на батче ≤ 1000 колонок — быстро (< 10 мс), NumPy, без I/O.

## Критерии приёмки

- [ ] `ColumnConfig` — frozen dataclass; `input_dim <= 0` → ValueError; `alpha ∉ [0, 1]` → ValueError
- [ ] `column_step()` — чистая: одинаковый вход → одинаковый выход, входные массивы не изменены
- [ ] `column_step()` — первый шаг из нулевого состояния: `e(0) = u(0)`
- [ ] `column_step()` — сходимость: стабильный вход → `e → 0` (в допуске)
- [ ] `column_step()` — `ValueError` при shape mismatch
- [ ] `CMCEnsemble.__init__()` — ValueError при пустом списке колонок
- [ ] `CMCEnsemble.__init__()` — ValueError при разных input_dim/state_dim у колонок
- [ ] `CMCEnsemble.step()` — корректная агрегация errors/states для N колонок
- [ ] `CMCEnsemble.step()` — `ValueError` при u.shape != (input_dim,)
- [ ] `CMCEnsemble.active` — корректный подсчёт по порогу `||e||² > threshold`
- [ ] `CMCEnsemble.reset()` — все состояния обнулены, active == 0
- [ ] Интеграционный тест cmc → energy: стабильный вход → errors → `F(t) → 0` через `FreeEnergyCalculator`
- [ ] Минимум 10 тестов: 5 core (config validation, purity, first step, convergence, shape), 5 shell (init validations ×2, step aggregation, active, reset) + 1 integration
- [ ] `ruff check` и `ruff format` проходят без ошибок
- [ ] mypy strict для `src/core/cmc/` не ругается

## Явно НЕ входит в скоуп (Phase 1)

- **Матрицы проекций/весов**: `state_dim == input_dim`, веса отсутствуют. Предиктивное кодирование с матрицами W — Фаза 2+
- **Обучение (Structure Learning)**: нет адаптации весов, нет schemas — Фаза 4
- **Батчинг SIMD**: `step()` — цикл по колонкам на NumPy; тензорная форма `[N, In, State]` одним вызовом — Фаза 3
- **Латеральное торможение / k-WTA**: это модуль `src/core/voting/`, не cmc
- **Проекции по специализациям**: колонка «тон» и колонка «ритм» в Фазе 1 считают одинаково — срезы входа — Фаза 2
- **Производство precision γ**: γ для energy задаётся вне cmc (см. Open Questions)
- **Связь с memory**: нет эмбеддингов, нет Episode — источник content/embedding всё ещё не заявлен
- **Принятие решений**: нет вызова LLM, нет триггеров, нет порога F(t) (это Фаза 2, см. energy SPEC)

## Open Questions

| Вопрос | Статус | Решение |
|--------|--------|---------|
| **Специализация**: строка-тег или enum? Влияет ли на проекцию входа уже в Фазе 1? | Решено | Строка-тег, в Фазе 1 не влияет на вычисления. Проекции (какая колонка какой срез входа видит) — Фаза 2, вместе с матрицами. |
| **Источник precision γ для energy**: кто его производит? | Решено | Манифест §3.Б: γ = обратная дисперсия входного сигнала. В Фазе 1 cmc НЕ производит γ — energy получает его из wiring/внешнего кода (как в текущем интеграционном тесте). Кандидат: cmc с оконной дисперсией e(t) — Фаза 2. |
| **Критерий активности колонки** | Решено | `||e(t)||² > threshold`, порог параметризуемый, дефолт 0.0 (любая ненулевая ошибка активна). Калибровка порога по телеметрии — Фаза 2. |
| **alpha по умолчанию** | Решено | 0.1 (период адаптации ≈ 10 тиков). Калибровка по сходимости на реальных входах — Фаза 2. |
| **Миграция существующего `Column` в `src/core/cmc/__init__.py`** | Решено | Класс `Column` (мутирует self.x/self.e — нарушает FC/IS) удаляется. `src/core/cmc/__init__.py` переписывается на re-export новых сущностей. `src/core/__init__.py` (единственный импортёр `Column`) обновляется одновременно — BACKLOG `[Phase3][cmc] Task 2`. |
| **state_dim vs input_dim** | Решено | В Фазе 1 `state_dim == input_dim` обязательно (нет матриц). Валидация в `ColumnConfig.__post_init__`. |

## Implementation Notes

1. **Структура файлов** (по ADR-0004 и паттерну energy): `models.py` (ColumnConfig, ColumnState, EnsembleOutput), `column.py` (column_step — Core), `ensemble.py` (CMCEnsemble — Shell), `__init__.py` (re-exports).
2. **Типизация numpy**: для векторов использовать `Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]` (идиома из `src/memory/serialize.py` — numpy-дженерики инвариантны).
3. **Связь с energy в wiring (Фаза 3, не сейчас)**: `EnsembleOutput.errors` имеет shape `[N, In]`, а `FreeEnergyCalculator.compute()` принимает векторы — ravel/агрегация — ответственность wiring, не cmc. В BACKLOG уже есть `[Phase3][cmc] Task 1` про `active_columns`.
4. **Чистота step()**: `column_step` не должен возвращать ссылки на переданные массивы — новые массивы через `u - prev.x` и т.п. Тест purity: изменить `u` после вызова — результат не меняется.
5. **Debug logging**: при срабатывании fail-fast валидаций — `logger.debug` с причиной (для Фазы 2 калибровки порогов).
