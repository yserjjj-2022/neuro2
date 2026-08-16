# SPEC.md — src/core/energy

## Назначение
Расчёт свободной энергии F(t), валентности (-dF/dt), аллостатического стресса и precision weighting γ для текущего состояния хоста. Экспортирует сырые метрики как наблюдаемые величины, не принимая решений.

## Публичный интерфейс

### FreeEnergyCalculator
```python
class FreeEnergyCalculator:
    """Калькулятор свободной энергии Фазы 1.
    
    Экспортирует сырые значения F(t), valence, stress, precision.
    Не принимает решений — только считает и возвращает.
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
            stress_decay: Коэффициент затухания стресса [0, 1]. Параметризуемый.
            gamma_base: Базовое значение precision weighting γ. Параметризуемый.
        """
        ...
    
    def compute(
        self,
        prediction_error: np.ndarray,
        precision: np.ndarray,
        prev_f: float,
    ) -> FreeEnergyResult:
        """Рассчитать F(t), valence, stress, gamma.
        
        Args:
            prediction_error: Вектор ошибки предсказания e(t) от колонок.
            precision: Вектор точности γ для каждого канала.
            prev_f: Значение F(t-1) для расчёта delta.
            
        Returns:
            FreeEnergyResult с полями: f, valence, stress, gamma.
        """
        ...
    
    def reset(self) -> None:
        """Сбросить внутреннее состояние (stress = 0, f = 0)."""
        ...
```

### FreeEnergyResult (dataclass)
```python
@dataclass(frozen=True)
class FreeEnergyResult:
    """Результат расчёта свободной энергии."""
    f: float                    # Свободная энергия F(t) ≥ 0
    valence: float              # Валентность -dF/dt
    allostatic_stress: float    # Интеграл F(t) по времени (затухающий)
    gamma: float                # Precision weighting γ
```

### EnergyObserver (shadow mode)
```python
class EnergyObserver:
    """Shadow-наблюдатель: логирует F(t) без принятия решений.
    
    Работает в Фазе 1 в режиме "shadow mode":
    - Считает, где сработал бы порог
    - Логирует, но ничего не вызывает
    - Ничего не блокирует
    """
    
    def __init__(
        self,
        calculator: FreeEnergyCalculator,
        f_threshold: Optional[float] = None,
    ) -> None:
        """
        Args:
            calculator: Calculator для расчёта метрик.
            f_threshold: Порог для анализа (None = shadow mode).
        """
        ...
    
    def observe(
        self,
        prediction_error: np.ndarray,
        precision: np.ndarray,
    ) -> FreeEnergyResult:
        """Наблюдать за состоянием: считать метрики и логировать.
        
        Args:
            prediction_error: Вектор ошибки предсказания e(t).
            precision: Вектор точности γ.
            
        Returns:
            FreeEnergyResult — сырые метрики без принятия решений.
        """
        ...
```

## Инварианты

1. **F(t) ≥ 0**: свободная энергия всегда неотрицательная (KL-дивергенция).
2. **Valence = -dF/dt**: валентность — производная F(t) со знаком минус.
3. **Stress монотонно затухает**: если F(t) = 0, stress не растёт (stress_decay < 1).
4. **γ > 0**: precision weighting всегда положительное.
5. **Non-blocking**: `compute()` не должен выполняться дольше 1 мс на батче ≤100 колонок.
6. **Stateless calculator**: `FreeEnergyCalculator.compute()` не изменяет внутреннее состояние (кроме stress, который является частью результата).

## Критерии приёмки

- [ ] `FreeEnergyCalculator.compute()` возвращает `FreeEnergyResult` с F(t) ≥ 0
- [ ] `valence` корректно рассчитывается как `-delta_f / dt`
- [ ] `allostatic_stress` монотонно затухает при F(t) = 0
- [ ] `gamma` всегда > 0
- [ ] `EnergyObserver.observe()` работает в shadow mode (f_threshold=None)
- [ ] Минимум 3 unit-теста: F(t) ≥ 0, valence sign, stress decay
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
| Формула stress decay | Решено | `stress = stress * stress_decay + F(t)` (затухающий интеграл) |
| Инициализация F(0) | Решено | `F(0) = 0` (система начинается с нуля) |
| Batch size для SIMD | Отложен | Фаза 3: батчинг колонок, пока одна колонка |
