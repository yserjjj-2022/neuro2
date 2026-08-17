# SPEC.md — src/core/attractors

## Назначение

Множественная устойчивая динамика выбора задачи (task attractors). Модуль берёт полный вектор активностей колонок (‖e‖²) и формирует **аттрактор задачи** — устойчивое представление, которое переживает шум входных сигналов и обеспечивает гистерезис (зависимость от истории, а не только от текущего входа).

Аттрактор — это не просто "победитель текущего тика". Это **бассейн притяжения** в пространстве задач: текущий mask удерживается несмотря на кратковременные флуктуации scores, пока конкурент не преодолеет барьер, зависящий от dwell time.

Фаза 2. Вход: полный вектор активностей `scores` (N,) float64 —‖e‖² по строкам errors из CMCEnsemble.

См. также:
- `src/core/voting/SPEC.md` — источник mask/scores (аттрактор получает scores напрямую)
- `BACKLOG [Phase2][cross-module] Threshold Calibration` — калибровка `active_threshold` (EMA ≠ 0)
- Манифест §3.В: "Task Goal — временные динамические аттракторы задач с TTL и приоритетом"

## Входные данные

```python
# Полный вектор активностей всех N колонок (из CMCEnsemble.step()):
scores: Vector  # (N,) float64 — активности ‖e‖² по строкам errors
```

## Публичный интерфейс

### TaskAttraction (frozen dataclass, Functional Core)

```python
@dataclass(frozen=True)
class TaskAttraction:
    """Снимок аттрактора задачи после tick().

    Аналог FreeEnergyResult (energy) и VotingResult (voting) — frozen
    dataclass как снимок состояния, неизменяемый после возврата.

    Attributes:
        mask: (N,) float64 — текущий аттрактор: маска активных колонок.
            Устойчив к шуму в scores — меняется только при преодолении
            порога переключения (см. Инвариант 3).
        history_size: int — сколько тиков подряд аттрактор не менялся.
            Используется для расчёта min_dwell (гистерезис + STP).
        scores: (N,) float64 — полный вектор активностей всех колонок.
            Записан из текущего входа для отладки и анализа конкурентности.
        converged: bool — True, если scores sufficiently small (EMA сходимость).
            При converged=True: mask может оставаться, но new_winner
            должен учитывать basin of attraction.
    """

    mask: Vector
    history_size: int
    scores: Vector
    converged: bool
```

### TaskAttractor (Imperative Shell)

```python
class TaskAttractor:
    """Imperative Shell: управляет динамикой аттрактора задачи.

    Functional Core / Imperative Shell (ADR-0004):
    - Core (compute_dwell, check_switch) — чистые функции
    - Shell (TaskAttractor) — хранит state, управляет переключениями

    Механизм переключения: SHORT-TERM PLASTICITY (STP)
    по Kubota & Aihara (2011). Не attractor-state itinerancy (стохастический
    переход по фиксированному ландшафту), а детерминированная реконфигурация:
    текущий аттрактор динамически усиливается при удержании и ослабевает
    при конкуренции. Это меняет сам ландшафт ям, а не просто толкает
    систему между фиксированными ямами.

    Ссылка: Kubota & Aihara, "Neural network model of short-term
    plasticity for working memory and attractor dynamics", 2011.
    """

    def __init__(
        self,
        n_tasks: int,
        base_dwell: int = 5,
        dwell_slope: float = 2.0,
        plasticity_gain: float = 0.1,
        basin_threshold: float = 0.15,
        convergence_threshold: float = 1e-8,
    ) -> None:
        """Инициализация аттрактора.

        Args:
            n_tasks: N — число колонок в ансамбле (размер mask).
            base_dwell: α — базовое время удержания в тиках.
                Минимальное время, которое аттрактор удерживается
                до первого разрешённого переключения.
            dwell_slope: β — чувствительность dwell time к разнице scores.
                Формула: min_dwell = α + β · (score_current - score_runner_up).
                При выигрыше (score_current > runner_up) dwell растёт (гистерезис).
                При проигрыше — падает.
            plasticity_gain: γ — прирост устойчивости за каждый тик удержания.
                Реализует STP (кратковременную пластичность): чем дольше
                удерживаем задачу, тем сильнее становимся (реконфигурация ландшафта).
            basin_threshold: ε — порог для basin of attraction.
                Если score_runner_up - score_current < ε, считаем,
                что мы в бассейне текущего аттрактора (устойчив к локальному шуму).
            convergence_threshold: τ — порог сходимости EMA.
                Если max(scores) < τ, все scores ~0 (система сходится).
                При converged=True: переключение запрещено, пока
                history_size < min_dwell (защита от флуктуаций нуля).

        Raises:
            ValueError: Если n_tasks < 2, base_dwell < 0, dwell_slope < 0,
                basin_threshold < 0, convergence_threshold <= 0.
        """
        if n_tasks < 2:
            raise ValueError("n_tasks < 2: не существует runner-up для сравнения")
        ...

    def tick(self, scores: Vector) -> TaskAttraction:
        """Один тик: обновить аттрактор на основе полного вектора активностей.

        Алгоритм:
        0. **Первый тик** (self._mask is None):
           - held_index = None
           - mask = one-hot для argmax(scores)
           - history_size = 0
           - converged = max(scores) < convergence_threshold
           - Return TaskAttraction(mask, history_size, scores.copy(), converged)
        1. Определить held_index — индекс победителя текущей маски (1.0 в mask).
        2. Найти score_runner_up = max(scores[i] for i in scores if i != held_index).
           (Лучший результат среди всех колонок, кроме удерживаемой — а не просто "топ-2").
        3. Найти score_current = scores[held_index].
        4. Проверить immediate_switch:
           - Если check_immediate_switch(score_current, score_runner_up):
             switch (новый mask = one-hot для argmax(scores), history_size = 0).
        5. Иначе (нет явного превосходства конкурента):
           a. Вычислить min_dwell = max(base_dwell, dwell_slope·Δ + gain·history).
           b. Если history_size < min_dwell: stay (гистерезис, история растёт).
           c. Иначе (history_size >= min_dwell):
              - `switch_now = (argmax(scores) != held_index) and not check_basin_stability(...)`
              - Если `switch_now`: switch (новый mask = one-hot для argmax(scores),
                history_size = 0)
              - Иначе (остаёмся — либо held и так топ-1, либо в бассейне):
                history_size += 1, stay
        6. converged = max(scores) < convergence_threshold.
        7. Return TaskAttraction(mask, history_size, scores.copy(), converged).

        Args:
            scores: Полный вектор активностей всех N колонок (‖e‖²).

        Returns:
            TaskAttraction — снимок аттрактора текущего тика.
        """
        ...

    def reset(self) -> None:
        """Сбросить аттрактор: mask = None, history = 0.
        
        Следующий tick() попадёт в ветку «первый тик» и заново выберет победителя.
        """
        ...

    @property
    def current_mask(self) -> Vector | None:
        """Текущий аттрактор (mask). None до первого tick()."""
        ...

    @property
    def history_size(self) -> int:
        """Сколько тиков подряд аттрактор не менялся."""
        ...
```

### compute_dwell (чистая функция, Functional Core)

```python
def compute_dwell(
    base_dwell: int,
    dwell_slope: float,
    plasticity_gain: float,
    score_current: float,
    score_runner_up: float,
    history_size: int,
) -> int:
    """Вычислить минимальное время удержания (гистерезис + STP).

    Формула:
        min_dwell = max(base, base + slope·Δ + gain·history)

    Семантика:
        - base_dwell — жёсткий пол удержания (не проседает). Гарантирует,
          что dwell-time механизм защищает от дребезга, а не маскирует его.
        - dwell_slope > 0: выигрыш продлевает удержание (гистерезис).
        - plasticity_gain > 0: чем дольше держусь, тем сильнее (STP).
        - Проигрыш НЕ снижает dwell ниже base — немедленное переключение
          управляется отдельным механизмом (see check_immediate_switch).

    Args:
        base_dwell: α — жёсткий пол удержания (не проседает).
        dwell_slope: β — прирост dwell при выигрыше (Δ > 0).
        plasticity_gain: γ — прирост устойчивости за каждый тик удержания (STP).
        score_current: Оценка текущей задачи (аттрактора).
        score_runner_up: Оценка претендента (топ-1 из scores, кроме held).
        history_size: Сколько тиков подряд аттрактор не менялся.

    Returns:
        min_dwell в тиках (>= base_dwell, округлено до ближайшего целого).

    Examples:
        # Выигрыш (+0.2 разницы): dwell растет (5.4 → 5)
        >>> compute_dwell(5, 2.0, 0.0, 0.5, 0.3, 0)
        5
        # Проигрыш (-0.3 разницы): dwell на полу (4.4 → 5, не 4!)
        >>> compute_dwell(5, 2.0, 0.0, 0.5, 0.8, 0)
        5
        # STP (history=5, gain=0.1): +0.5 к dwell
        >>> compute_dwell(5, 0.0, 0.1, 0.5, 0.5, 5)
        6
    """
    delta = score_current - score_runner_up
    raw_dwell = base_dwell + dwell_slope * delta + plasticity_gain * history_size
    return max(base_dwell, round(raw_dwell))
```

### check_basin_stability (чистая функция, Functional Core)

```python
def check_basin_stability(
    basin_threshold: float,
    score_current: float,
    score_runner_up: float,
    history_size: int,
) -> bool:
    """Проверить, находится ли система в бассейне текущего аттрактора.

    Для локального шума: если проигрыш не превышает порог, остаемся.
    При history_size > 0 аттрактор сильнее (STP), порог эффективный растет.

    Args:
        basin_threshold: ε — базовый порог бассейна притяжения.
        score_current: Оценка текущей задачи.
        score_runner_up: Оценка претендента.
        history_size: Текущая история удержания (влияет на STP).

    Returns:
        True, если переключение НЕ требуется (в бассейне).
    """
    # Чем больше history_size, тем сложнее выбить из аттрактора (STP)
    effective_threshold = basin_threshold + 0.05 * history_size
    delta = score_current - score_runner_up
    # Остаемся, если проигрываем не более чем на эффективный порог
    return delta > -effective_threshold
```

### check_immediate_switch (чистая функция, Functional Core)

```python
def check_immediate_switch(
    score_current: float,
    score_runner_up: float,
    dominance_threshold: float = 0.3,
) -> bool:
    """Проверить, есть ли явное превосходство конкурента.

    При сильном проигрыше (Δ < -dominance_threshold) разрешаем немедленное
    переключение, игнорируя min_dwell — это не дребезг, а осознанная уступка.
    Отличается от dwell-mеханизма: dwell защищает от быстрых флип-флопов,
    а immediate_switch разрешает переключение, когда конкурент ЯВНО выигрывает.

    Args:
        score_current: Оценка текущей задачи.
        score_runner_up: Оценка претендента.
        dominance_threshold: Порог явного превосходства (Δscore < -ε).

    Returns:
        True, если конкурент явно выигрывает (переключаемся немедленно).
    """
    return (score_runner_up - score_current) > dominance_threshold
```

## Инварианты

1. **FC/IS**: `compute_dwell` и `check_basin_stability` — чистые функции (одинаковый вход → одинаковый выход). `TaskAttractor` — единственный владелец состояния (mask, history).

2. **Basin of attraction (устойчивость к локальному шуму)**: при `Δscore < basin_threshold` аттрактор НЕ переключается, даже если новый победитель k-WTA изменился. Это защита от точечных выбросов в scores (локальный шум), где доминирование определяется структурой динамических переходов, а не размером бассейна.

3. **Hysteresis (гистерезис)**: точка переключения зависит от истории системы, а не только от текущего входа. `min_dwell = max(base, slope·Δ + gain·history)` обеспечивает доказуемое условие стабильности:
   - При выигрыше (Δ > 0): dwell растёт выше base (продолжаем удержание).
   - При проигрыше (Δ < 0): dwell остаётся на base (защита от дребезга).
   - При явном превосходстве конкурента: `check_immediate_switch` разрешает переключение независимо от dwell — это не дребезг, а осознанная уступка.
   Это маркер дискретных аттракторных состояний (не плавное следование за стимулом).

4. **Convergence safety**: при `max(scores) < convergence_threshold` (EMA сходится к нулю) переключение запрещено до `history_size >= min_dwell`. Решает проблему float64-артефакта: EMA никогда не достигает точного нуля, поэтому без dwell-защаты аттрактор будет флуктуировать между нулевыми scores.

5. **STP mechanism (не itinerancy)**: аттрактор не просто "перескакивает" между фиксированными ямами. Текущий mask усиливается при удержании (через увеличение `history_size`, которое влияет на будущие `min_dwell`). Это реконфигурация ландшафта, а не стохастический переход.

6. **Dwell time — параметр, не константа**: `min_dwell` всегда вычисляется как `f(Δscore)`, никогда не хардкодится. Решает ту же проблему, что `active_threshold` в energy/cmc/voting: магическое число без обоснованной калибровки.

7. **Non-blocking**: `tick()` на батче ≤ 1000 колонок — быстро (< 1 мс), NumPy, без I/O.

## Критерии приёмки

- [ ] `compute_dwell()` — чистая: одинаковый вход → одинаковый выход
- [ ] `compute_dwell()` — `Δscore = 0` → `min_dwell == base_dwell`
- [ ] `compute_dwell()` — `Δscore > 0` → `min_dwell > base_dwell`
- [ ] `compute_dwell()` — `Δscore < 0` → `min_dwell == base_dwell` (проигрыш не снижает dwell)
- [ ] `compute_dwell()` — base_dwell — жёсткий пол: min_dwell >= base_dwell всегда
- [ ] `check_immediate_switch()` — большой Δscore (конкурент явнее) → True
- [ ] `check_immediate_switch()` — малый Δscore → False
- [ ] `check_basin_stability()` — `Δscore < basin_threshold` → True (в бассейне)
- [ ] `check_basin_stability()` — `Δscore >= basin_threshold` → False (вне бассейна)
- [ ] `check_basin_stability()` — разные индексы победителей, малый Δscore → True (локальный шум)
- [ ] `TaskAttractor.tick()` — immediate_switch: явное превосходство конкурента → switch
- [ ] `TaskAttractor.tick()` — первый тик: mask = one-hot, history_size = 0
- [ ] `TaskAttractor.tick()` — stable input: mask не меняется, history_size растёт
- [ ] `TaskAttractor.tick()` — switch: новый mask = one-hot для current_winner_index, history_size = 0
- [ ] `TaskAttractor.__init__()` — `n_tasks < 2` → ValueError
- [ ] `TaskAttractor.tick()` — convergence: при scores < τ, switch запрещён до history_size >= min_dwell
- [ ] `TaskAttractor.tick()` — cool-down: history_size < min_dwell → mask не меняется
- [ ] `TaskAttractor.reset()` — mask = None, history = 0
- [ ] `TaskAttractor.tick()` — reset() затем tick(): mask = one-hot по argmax, history = 0 (как новый объект)
- [ ] Интеграционный тест attractors ← voting: полный цикл cmc → voting → attractor
- [ ] Минимум 19 тестов: 10 core (compute_dwell ×5, check_immediate_switch ×2, check_basin_stability ×3), 8 shell (init, tick_first, tick_stable, tick_switch, convergence, immediate_switch, basin_stability, reset, tick_after_reset, current_mask_none_before)
- [ ] `ruff check` и `ruff format` проходят без ошибок
- [ ] mypy strict для `src/core/attractors/` не ругается

## Явно НЕ входит в скоуп (Phase 2)

- **Reflex-сигналы** не проходят через TaskAttractor — см. [Phase3][sensory] Reflex path в BACKLOG. Dwell-логика принципиально не предназначена для критических/интероцептивных сигналов.
- **TTL задач**: время жизни задачи до автоматического истечения — Фаза 3
- **Приоритеты задач**: иерархия нескольких одновременных задач — Фаза 3
- **Эпистемический драйв**: задача выбирается по максимальной неопределённости — Фаза 3
- **Joint Agency**: совместные целевые аттракторы для ToM — Фаза 3
- **Structure Learning (Schemas)**: обобщение повторяющихся паттернов — Фаза 4
- **Консолидация во сне**: active pruning эпизодов — Фаза 4
- **Связь с memory**: нет эмбеддингов, нет Episode — источник content/embedding не заявлен
- **Масштабирование на батч**: один TaskAttractor на одну задачу; N параллельных аттракторов — Фаза 3

## Open Questions

| Вопрос | Статус | Решение |
|--------|--------|---------|
| **base_dwell по умолчанию** | Открыто | 5 тиков — рабочая константа для тестов. Калибровка по реальным входам — Фаза 3. |
| **dwell_slope по умолчанию** | Открыто | 2.0 — чувствительность к Δscore. Нужно проверить на данных: при каких Δscore dwell становится существенным (> 10 тиков). |
| **basin_threshold по умолчанию** | Открыто | 0.15 — порог для локального шума. Зависит от масштаба scores (EMA-активности колонок). Калибровка по distribution Δscore — Фаза 3. |
| **Глобальный vs локальный шум** | Решено | Фаза 2: только локальный шум (точечные выбросы в scores). Глобальное дрожание всех scores — Фаза 3 (проверка размера бассейна). |
| **STP vs itinerancy** | Решено | Фаза 2: STP (детерминированная реконфигурация ландшафта). Attractor-state itinerancy — Фаза 3 (сравнительный анализ, скрытые марковские модели). |
| **Convergence threshold** | Решено | 1e-8 — согласовано с `[Phase2][cross-module] Threshold Calibration`. Используется во всех трёх модулях (cmc, voting, energy). |
| **Dominance threshold** | Открыто | 0.3 — порог для `check_immediate_switch`. При каком Δscore конкурент становится "явным"? Калибровка по distribution Δscore — Фаза 3. |

## Implementation Notes

1. **Структура файлов** (по ADR-0004 и паттерну voting/energy):
   - `models.py` — `TaskAttraction` (frozen dataclass)
   - `compute.py` — `compute_dwell`, `check_basin_stability`, `check_immediate_switch` (Core)
   - `manager.py` — `TaskAttractor` (Shell)
   - `__init__.py` — re-export
   - `SPEC.md` — этот файл
   - `README.md` — документация

2. **Зависимости**:
   - Внутренние: `src/core/voting` — `VotingResult` не требуется (attractor получает полный вектор scores напрямую из wiring)
   - Внешние: `numpy` — mask operations, argmax
   - Стандартная библиотека: `dataclasses`, `typing`

3. **Связь с wiring (Фаза 3, не сейчас)**: `TaskAttractor.tick()` вызывается из `CMCPipeline.tick()` после строки `self.voting.vote(activities)`. Порядок вызовов в wiring.py — просто последовательность строк кода, а не логическая зависимость данных: voting и attractors — параллельные, независимые потребители одного и того же вектора activities (см. Note №9).

4. **Типизация numpy**: `Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]` (идиома из `src/core/voting/models.py`). `IndexVector = np.ndarray[Any, np.dtype[np.intp]]`.

5. **Purity**: `compute_dwell` и `check_basin_stability` не мутируют входные массивы. Тест purity: изменить входной массив после вызова — результат не меняется. Кроме того, `TaskAttraction.scores` хранит копию `scores.copy()`, а не ссылку на переданный вызывающим кодом массив — иначе, если wiring.py мутирует свой буфер activities между тиками (для переиспользования памяти), исторический снимок в TaskAttraction задним числом изменится, что нарушит purity-гарантию.

6. **Debug logging**: при переключении аттрактора — `logger.debug` с Δscore, min_dwell, reason (basin / convergence / switch) — для калибровки порогов в Фаза 3.

7. **Runner-up calculation**: `score_runner_up = max(scores[i] for i in range(len(scores)) if i != held_index)`. Это не просто "топ-2" в отсортированном списке, а лучший результат среди всех колонок, кроме удерживаемой. Технически совпадает с "топ-2", когда held действительно топ-1, но расходится, когда held проигрывает — наивная реализация через `np.argsort(scores)[-2:]` без учёта `held_index` даст неверное значение в этом случае.

8. **Dynamic dwell check**: `dwell_remaining` не хранится как отдельное состояние. На каждом тике сравнивается `history_size < compute_dwell(...)`, пересчитывая порог заново из актуальных scores. Это чище с точки зрения FC/IS: вся логика принятия решения — в Core, Shell только хранит минимально необходимое состояние (mask, history), а не кэширует производную величину, которая может протухнуть.

9. **Архитектурная развязка с voting**: `TaskAttractor.tick()` читает scores напрямую из CMCEnsemble, минуя `VotingManager`. `VotingManager.vote()` и `TaskAttractor.tick()` — параллельные, независимые потребители одного и того же вектора activities. Это снимает блокировку k=1 и позволяет attractor видеть полную картину конкурентности. Комментарий в wiring.py про "интерфейс для будущих аттракторов" устарел — voting существует для собственной телеметрии/логирования, не как звено пайплайна к аттракторам.
