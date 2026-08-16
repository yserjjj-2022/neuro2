# SPEC.md — src/core/energy

## Назначение
Расчёт свободной энергии F(t), валентности (-dF/dt), аллостатического стресса и precision weighting γ для текущего состояния хоста. Экспортирует сырые метрики как наблюдаемые величины, не принимая решений.

## Публичный интерфейс

### Формула свободной энергии

```
F(t) = 0.5 · Σᵢ γᵢ · e(t)ᵢ²
```

где:
- `e(t)` — вектор ошибки предсказания (prediction_error)
- `γ` — вектор точности (precision)
- Сумма по всем элементам вектора

**Пограничные случаи:**
- Если `prediction_error.shape != precision.shape` → `ValueError`
- Если векторы пустые (0 колонок) → `F(t) = 0.0`

### FreeEnergyResult (dataclass)

```python
@dataclass(frozen=True)
class FreeEnergyResult:
    """Результат расчёта свободной энергии."""
    f: float                    # Свободная энергия F(t) ≥ 0
    valence: float              # Валентность -dF/dt
    allostatic_stress: float    # Интеграл F(t) по времени (затухающий)
    gamma: float                # Precision weighting γ
    would_trigger: bool | None  # При f_threshold=None — None (shadow mode)
```

### FreeEnergyCalculator (fully stateless)

```python
class FreeEnergyCalculator:
    """Чистый калькулятор свободной энергии — полностью stateless.
    
    Соответствует паттерну Functional Core:
    - Все вычисления — чистые функции без побочных эффектов
    - Состояние (prev_f, prev_stress) хранится вызывающим кодом
    - Один и тот же input → один и тот же output без зависимости от порядка вызовов
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        stress_decay: float = 0.99,
        gamma_base: float = 1.0,
    ) -> None:
        """Инициализация калькулятора.
        
        Args:
            dt: Шаг интегрирования (секунды). Параметризуемый.
            stress_decay: Коэффициент затухания стресса [0, 1). Параметризуемый.
            gamma_base: Базовое значение precision weighting γ. Параметризуемый.
        """
        ...
    
    def compute(
        self,
        prediction_error: np.ndarray,
        precision: np.ndarray,
        prev_f: float,
        prev_stress: float,
    ) -> FreeEnergyResult:
        """Рассчитать F(t), valence, stress, gamma.
        
        Чистая функция: не изменяет внутреннее состояние.
        
        Args:
            prediction_error: Вектор ошибки предсказания e(t).
            precision: Вектор точности γ для каждого канала.
            prev_f: Значение F(t-1).
            prev_stress: Значение allostatic_stress(t-1).
            
        Returns:
            FreeEnergyResult с полями: f, valence, stress, gamma, would_trigger=None.
            
        Raises:
            ValueError: Если prediction_error.shape != precision.shape.
            
        Formula:
            F(t) = 0.5 · Σᵢ γᵢ · e(t)ᵢ²
            valence = -(F(t) - prev_f) / dt
            stress = prev_stress * stress_decay + F(t)
            gamma = np.mean(precision)
        """
        ...
    
    # reset() УДАЛЁН: stateless-архитектура не требует сброса
```

### EnergyObserver (DI через sink)

```python
class EnergyObserver:
    """Shadow-наблюдатель: логирует F(t) без принятия решений.
    
    Functional Core / Imperative Shell:
    - Core (calculator) — чистая функция, тестируется без I/O
    - Shell (observer) — инъекция sink, можно мокать в тестах
    - В проде: sink = telemetry_logger.log_event
    - В тестах: sink = list.append
    """
    
    def __init__(
        self,
        calculator: FreeEnergyCalculator,
        sink: Callable[[FreeEnergyResult], None] | None = None,
    ) -> None:
        """
        Args:
            calculator: Calculator для расчёта метрик.
            sink: Необязательная функция записи. Если None — observe() 
                  возвращает результат без записи.
        """
        ...
    
    def observe(
        self,
        prediction_error: np.ndarray,
        precision: np.ndarray,
    ) -> FreeEnergyResult:
        """Наблюдать за состоянием: считать метрики, записать через sink.
        
        Args:
            prediction_error: Вектор ошибки предсказания e(t).
            precision: Вектор точности γ.
            
        Returns:
            FreeEnergyResult — сырые метрики без принятия решений.
            Поле would_trigger устанавливается в True/False в зависимости
            от порога, но не вызывает действий.
        """
        ...
```

## Инварианты

1. **F(t) ≥ 0**: свободная энергия всегда неотрицательная (KL-дивергенция, квадратичная форма).
2. **Valence = -dF/dt**: валентность — производная F(t) со знаком минус.
3. **Stress монотонно затухает**: если F(t) = 0, stress не растёт (stress_decay < 1).
4. **γ > 0**: precision weighting всегда положительное.
5. **Non-blocking**: `compute()` не должен выполняться дольше 1 мс на батче ≤100 колонок.
6. **Fully stateless**: `FreeEnergyCalculator.compute()` не изменяет внутреннее состояние. Все временные переменные (prev_f, prev_stress) передаются явно.
7. **Shape validation**: `ValueError` при несовпадении размерностей prediction_error и precision.

## Критерии приёмки

- [ ] `FreeEnergyCalculator.compute()` — полностью stateless, prev_stress — явный параметр
- [ ] Формула F(t) = 0.5 · Σ γᵢ · e(t)ᵢ² реализована верно
- [ ] `ValueError` при несовпадении размерностей prediction_error и precision
- [ ] Пустые векторы → F(t) = 0.0
- [ ] `valence` корректно рассчитывается как `-(f - prev_f) / dt`
- [ ] `allostatic_stress` монотонно затухает при F(t) = 0
- [ ] `gamma` всегда > 0
- [ ] `EnergyObserver` тестируется без файловой системы (sink=list.append)
- [ ] Минимум 4 unit-теста: F(t) formula, shape validation, stress decay, valence sign
- [ ] `ruff check` и `ruff format` проходят без ошибок
- [ ] mypy strict не ругается

## Явно НЕ входит в скоуп

- **Принятие решений**: нет вызова LLM, нет изменения поведения
- **Калибровка порога**: F(t) threshold — отдельная задача Фазы 2
- **Связь с колонками**: нет прямого доступа к CMC, только через e(t) и γ
- **Визуализация**: нет графиков, нет dashboard
- **Structure Learning**: нет обобщения паттернов, нет schemas
- **Ночной сон**: нет active pruning, нет consolidation

## Open Questions

| Вопрос | Статус | Решение |
|--------|--------|---------|
| **Порог F(t) для event-triggered вызова LLM** | **Отложен до Фазы 2** | **В Фазе 1: EnergyObserver в shadow mode — только логирует, ничего не триггерит. Калибровка порога по перцентилю — Фаза 2, после сбора статистики.** |
| Формула stress decay | Решено | `stress = prev_stress * stress_decay + F(t)` (затухающий интеграл) |
| Инициализация F(0), stress(0) | Решено | `F(0) = 0`, `stress(0) = 0` (система начинается с нуля) |
| Batch size для SIMD | Отложен | Фаза 3: батчинг колонок, пока одна колонка |
| would_trigger в результате | Решено | `would_trigger: bool | None` — None в shadow mode, True/False при указании порога |
