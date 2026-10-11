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

### attention_gate / apply_attention (чистые функции, Functional Core, S4)

```python
def attention_gate(gamma, gamma_ref=1.0, floor=0.0) -> Vector:
    """a(γ) = floor + (1 - floor)·γ/(γ + gamma_ref), ∈ [floor, 1)."""


def apply_attention(u, weights) -> Vector:
    """u_eff = u · a (покомпонентно, без мутации входа)."""
```

Пред-колоночный барьер внимания (манифест §2, ADR-0005 §2): доверие каналу
γ управляет прохождением входа. Низкая γ (шумный канал) аттенюируется,
высокая — проходит. Применяется в host loop при `attention_gate=True`;
иначе `u_eff == u` (контур S1–S3).

### ColumnConfig (frozen dataclass, Functional Core)

```python
@dataclass(frozen=True)
class ColumnConfig:
    """Конфигурация одной колонки. Только параметры — состояние не здесь."""

    input_dim: int  # Размерность входа L4
    state_dim: int  # Размерность состояния L5/6
    specialization: str = (
        "general"  # Тег: "tone", "rhythm", "meaning", "tom", "mcp", ...
    )
    alpha: float = 0.1  # Скорость обновления состояния, α ∈ [0, 1]
    reads: tuple[str, ...] | None = None  # Имена сегментов шины (ADR-0013)
```

Note:
- `specialization` — тег-идентификатор колонки (в выборе задачи — по тегу).
- `reads` — **имена** сегментов шины, которые колонка читает (ADR-0013).
  `None` → весь вход (legacy, `m ≡ 1`). Пустой кортеж → `ValueError` (колонка
  без входа). Имена разрешаются в бинарную маску при сборке ансамбля (Shell),
  не в конфиге. Проекция — **маска полного вектора** (инвариант `[N, In, State]`
  сохранён), не подвектор.
- `state_dim == input_dim` в Фазе 1 (матрицы проекций нет). Несовпадение → `ValueError` в конфиге.
- Дефолты только для `alpha`, `specialization`, `reads`; размерности — обязательные аргументы (без неявных значений).

### ColumnState (frozen dataclass, Functional Core)

```python
@dataclass(frozen=True)
class ColumnState:
    """Иммутабельный снимок состояния колонки: x(t) и e(t)."""

    x: np.ndarray  # Состояние L5/6, shape == (state_dim,)
    e: np.ndarray  # Ошибка предсказания L2/3, shape == (state_dim,)
```

Note:
- Не мутируется `column_step` — функция чистая, `prev` передаётся явно.
- Аналогия: `Episode` (memory) и `FreeEnergyResult` (energy) — frozen dataclass как снимок.

### column_step (чистая функция, Functional Core)

```python
def column_step(
    cfg: ColumnConfig,
    u: np.ndarray,
    prev: ColumnState,
    mask: np.ndarray | None = None,
) -> ColumnState:
    """Один шаг динамики колонки (с проекцией reads — маска ADR-0013).

    Чистая функция: не мутирует prev и u, не хранит состояние.
    Состояние передаётся явно — вызывающий код (ensemble) владеет им.

    Args:
        cfg: Конфигурация колонки.
        u: Вход L4, shape == (input_dim,).
        prev: Состояние на предыдущем шаге x(t-1), e(t-1).
        mask: Бинарная маска проекции reads, shape == (input_dim,). None →
            весь вход (legacy, m ≡ 1).

    Returns:
        Новый ColumnState: x(t), e(t).

    Raises:
        ValueError: Если u.shape != prev.x.shape (или != cfg.input_dim).

    Formula (mask == None → m ≡ 1):
        x(t) = x(t-1) + alpha · m ⊙ (u(t) − x(t-1))
        e(t) = m ⊙ (u(t) − x(t))
    """
    ...
```

Note:
- Первый шаг: `prev = ColumnState(x=zeros(input_dim), e=zeros(input_dim))` — инициализация вызывающим кодом (ensemble), не внутри функции.
- Функция возвращает новый объект; контракт: входные массивы не изменяются.
- Маска означает «не наблюдается», а не «наблюдается ноль» (ADR-0013 §2):
  нечитаемые координаты не обновляют `x` и не создают `e`. Подача `m⊙u` в
  прежний фильтр дала бы фиктивную ошибку от нуля — это исключено.
- `mask is None` → формулы тождественны прежним (`m ≡ 1`), legacy-контракт.

### EnsembleOutput (frozen dataclass, Imperative Shell)

```python
@dataclass(frozen=True)
class EnsembleOutput:
    """Снимок всего ансамбля после step()."""

    errors: np.ndarray  # e(t) всех колонок, shape == (N, input_dim)
    states: np.ndarray  # x(t) всех колонок, shape == (N, state_dim)
    active: int  # Число активных колонок (для telemetry active_columns)
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
        active_threshold: float = 1e-8,
        segment_dims: Mapping[str, int] | None = None,
    ) -> None:
        """Создание ансамбля: валидация, разрешение reads → маски, инициализация.

        Args:
            columns: Конфигурации колонок. Все должны иметь одинаковые
                input_dim и state_dim (единый тензор [N, In, State],
                конституция §2.1).
            active_threshold: Колонка активна, если ||e(t)||² > threshold.
                Дефолт 1e-8: EMA never converges to exact 0.0 in float64,
                so 0.0 would produce false-positive active_columns.
            segment_dims: Карта входных сегментов ансамбля ``{имя: длина}`` в
                порядке укладки (для разрешения ``reads``; ADR-0013). None →
                ``reads`` у колонок должны быть None.

        Raises:
            ValueError: Если список пуст; input_dim/state_dim различаются;
                размерности <= 0; или ``reads`` не разрешаются (неизвестный
                сегмент, пустой reads, повтор имени, reads без segment_dims).
        """
        ...

    def step(self, u: np.ndarray) -> EnsembleOutput:
        """Один тик: прогнать все колонки через column_step (со своей маской).

        Args:
            u: Вход ансамбля L4, shape == (input_dim,). Один вектор,
                общий для всех колонок; каждая читает свою проекцию reads
                (маска полного вектора, ADR-0013).

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

    @property
    def resolved_reads(self) -> tuple[tuple[str, ...], ...]:
        """Фактические reads колонок (наблюдаемость, ADR-0013 §8).

        Кортеж по колонкам; каждая — имена разрешённых сегментов в порядке
        карты (или все сегменты, если ``reads=None``).
        """
        ...
```

**Проекции (reads → маска, ADR-0013).** При сборке `CMCEnsemble` разрешает
`reads` каждой колонки в бинарную маску `m_j` (`input_dim`) по `segment_dims`;
маски неизменны в прогоне. Разрешение имён — чистая функция
`resolve_reads(reads, segment_dims)` (`projection.py`, Core): неизвестный
сегмент / пустой `reads` / повтор имени / `reads` без карты → `ValueError` до
первого тика. Наблюдаемость — `resolved_reads` (ансамбль) и лог при сборке loop.

**Правила агрегации (ADR-0013 §4).** Активность колонки — `Σ e²` по
координатам (legacy, не среднее); глобальный FFF — сумма по всем колонкам;
k-WTA/аттрактор — по `Σ e²` (legacy). Маскированные координаты дают точный 0 и
не участвуют ни в активности, ни в тегах, ни в энергии.

## Инварианты

1. **FC/IS**: `column_step` — чистая функция (одинаковый `(cfg, u, prev, mask)` → одинаковый результат; входные массивы не мутируются). `CMCEnsemble` — единственный владелец состояний.
2. **Сходимость**: при стабильном `u` ошибка `e(t) → 0` на читаемых координатах (колонка «привыкает» к входу).
3. **Локальные обновления**: состояние колонки зависит только от её собственного входа (проекции `reads`) и состояния (конституция §2.1 — никакого обмена между колонками, никакого backprop через ансамбль).
4. **Единые размерности**: все колонки ансамбля имеют одинаковые `input_dim`/`state_dim` — единый тензор `[N, In, State]`. Проекция — **маска полного вектора**, не подвектор.
5. **Fail-fast**: любые несоответствия размерностей, `α ∉ [0, 1]` и неразрешимые `reads` → `ValueError` до вычислений (не клиппинг, не молчаливая коррекция).
6. **Параметризуемость**: `alpha`, `active_threshold`, `reads` — параметры конструкторов, не хардкод (конституция §2.2).
7. **Non-blocking**: `step()` на батче ≤ 1000 колонок — быстро (< 10 мс), NumPy, без I/O.
8. **Маска «не наблюдается»**: нечитаемые координаты не обновляют `x`, не создают `e`, не участвуют в активности/тегах/энергии (ADR-0013 §2). `reads=None` → `m ≡ 1` (legacy).

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
- [x] `resolve_reads()` — имена → маска; неизвестный/пустой/повтор/без карты → ValueError
- [x] `column_step(mask)` — маска «не наблюдается»: нечитаемые координаты не обновляют x и e
- [x] `column_step(mask=None)` — legacy-тождество (`m ≡ 1`)
- [x] `CMCEnsemble(segment_dims)` — избирательность: возмущение сегмента меняет только читающие колонки
- [x] `CMCEnsemble.resolved_reads` — наблюдаемость фактических reads
- [ ] Интеграционный тест cmc → energy: стабильный вход → errors → `F(t) → 0` через `FreeEnergyCalculator`
- [ ] Минимум 10 тестов: 5 core (config validation, purity, first step, convergence, shape), 5 shell (init validations ×2, step aggregation, active, reset) + 1 integration
- [ ] `ruff check` и `ruff format` проходят без ошибок
- [ ] mypy strict для `src/core/cmc/` не ругается

## Явно НЕ входит в скоуп (Phase 1)

- **Матрицы проекций/весов**: `state_dim == input_dim`, веса отсутствуют. Предиктивное кодирование с матрицами W — Фаза 2+
- **Обучение (Structure Learning)**: нет адаптации весов, нет schemas — Фаза 4
- **Батчинг SIMD**: `step()` — цикл по колонкам на NumPy; тензорная форма `[N, In, State]` одним вызовом — Фаза 3
- **Латеральное торможение / k-WTA**: это модуль `src/core/voting/`, не cmc
- **Проекции по специализациям (матрицы весов)**: проекция — бинарная маска reads (ADR-0013); обучаемых матриц проекций нет — Фаза 2+
- **Нормировка активностей (mean по reads)**: активность — `Σ e²` (legacy); переход на среднее и его влияние на argmax при разной ширине — отдельный шаг
- **Разные alpha (временные шкалы)**: тривиатор специализации по времени — отдельно от reads
- **Единицы/масштаб каналов**: нормализация диапазонов — отдельный шаг
- **Производство precision γ**: γ для energy задаётся вне cmc (см. Open Questions)
- **Связь с memory**: нет эмбеддингов, нет Episode — источник content/embedding всё ещё не заявлен
- **Принятие решений**: нет вызова LLM, нет триггеров, нет порога F(t) (это Фаза 2, см. energy SPEC)

## Open Questions

| Вопрос | Статус | Решение |
|--------|--------|---------|
| **Специализация**: строка-тег или enum? Влияет ли на проекцию входа? | Решено | Строка-тег (идентификатор колонки в выборе задачи). Проекция входа — **именованные reads** сегментов → бинарная маска при сборке (ADR-0013); обучаемых матриц нет. |
| **Форма проекции: маска vs подвектор** | **Решено (2026-10-11)** | Маска полного вектора (ADR-0013 §2–3): сохраняет инвариант `[N, In, State]`; семантика «не наблюдается» (нечитаемые координаты не дают фиктивной ошибки). Подвектор отвергнут. |
| **Метрика активности: sum vs mean по reads** | **Решено (2026-10-11)** | `Σ e²` (legacy, ADR-0013 §4). При равной ширине reads sum/mean эквивалентны по ранжированию; переход на mean — отдельный шаг с перепечаткой отпечатков. |
| **Источник precision γ для energy**: кто его производит? | Решено | Манифест §3.Б: γ = обратная дисперсия входного сигнала. В Фазе 1 cmc НЕ производит γ — energy получает его из wiring/внешнего кода (как в текущем интеграционном тесте). Кандидат: cmc с оконной дисперсией e(t) — Фаза 2. |
| **Критерий активности колонки** | **Решено (2026-08-17)** | `||e(t)||² > threshold`, порог параметризуемый. **Дефолт 1e-8** (не 0.0): EMA never converges to exact 0.0 в float64, так что 0.0 давала бы ложноположительные active_columns при любой длительной сходимости. Калибровка порога по телеметрии — Фаза 2. |
| **alpha по умолчанию** | Решено | 0.1 (период адаптации ≈ 10 тиков). Калибровка по сходимости на реальных входах — Фаза 2. |
| **Миграция существующего `Column` в `src/core/cmc/__init__.py`** | Решено | Класс `Column` (мутирует self.x/self.e — нарушает FC/IS) удаляется. `src/core/cmc/__init__.py` переписывается на re-export новых сущностей. `src/core/__init__.py` (единственный импортёр `Column`) обновляется одновременно — BACKLOG `[Phase3][cmc] Task 2`. |
| **state_dim vs input_dim** | Решено | В Фазе 1 `state_dim == input_dim` обязательно (нет матриц). Валидация в `ColumnConfig.__post_init__`. |

## Implementation Notes

1. **Структура файлов** (по ADR-0004 и паттерну energy): `models.py` (ColumnConfig, ColumnState, EnsembleOutput), `column.py` (column_step — Core), `ensemble.py` (CMCEnsemble — Shell), `__init__.py` (re-exports).
2. **Типизация numpy**: для векторов использовать `Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]` (идиома из `src/memory/serialize.py` — numpy-дженерики инвариантны).
3. **Связь с energy в wiring (Фаза 3, не сейчас)**: `EnsembleOutput.errors` имеет shape `[N, In]`, а `FreeEnergyCalculator.compute()` принимает векторы — ravel/агрегация — ответственность wiring, не cmc. В BACKLOG уже есть `[Phase3][cmc] Task 1` про `active_columns`.
4. **Чистота step()**: `column_step` не должен возвращать ссылки на переданные массивы — новые массивы через `u - prev.x` и т.п. Тест purity: изменить `u` после вызова — результат не меняется.
5. **Debug logging**: при срабатывании fail-fast валидаций — `logger.debug` с причиной (для Фазы 2 калибровки порогов).
