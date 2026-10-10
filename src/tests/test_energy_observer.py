import numpy as np
import pytest

from src.core.energy.calculator import FreeEnergyCalculator
from src.core.energy.models import FreeEnergyResult
from src.core.energy.observer import EnergyObserver


@pytest.fixture
def observer() -> EnergyObserver:
    calc = FreeEnergyCalculator()
    return EnergyObserver(calc)


def test_observer_no_sink(observer: EnergyObserver) -> None:
    """observe() возвращает результат без sink."""
    error = np.array([1.0])
    precision = np.array([1.0])

    result = observer.observe(error, precision, dt=0.01)

    assert isinstance(result, FreeEnergyResult)


def test_observer_with_sink(observer: EnergyObserver) -> None:
    """sink вызывается с результатом."""
    log: list[FreeEnergyResult] = []
    observer.sink = log.append
    error = np.array([1.0])
    precision = np.array([1.0])

    observer.observe(error, precision, dt=0.01)

    assert len(log) == 1
    assert isinstance(log[0], FreeEnergyResult)


def test_observer_passes_importance(observer: EnergyObserver) -> None:
    """observe() прокидывает importance в калькулятор (взвешенный F)."""
    error = np.array([1.0, 1.0])
    precision = np.array([1.0, 1.0])

    result = observer.observe(
        error, precision, dt=0.01, importance=np.array([2.0, 2.0])
    )

    assert result.f == pytest.approx(2.0)  # 0.5 * (2 + 2)


def test_observer_maintains_state(observer: EnergyObserver) -> None:
    """Два последовательных observe() корректно передают состояние."""
    error1 = np.array([1.0])
    precision1 = np.array([1.0])

    result1 = observer.observe(error1, precision1, dt=0.01)

    error2 = np.array([1.0])
    precision2 = np.array([1.0])

    result2 = observer.observe(error2, precision2, dt=0.01)

    # Состояние обновлено после второго вызова
    assert observer.state.f == result2.f
    assert observer.state.stress == result2.allostatic_stress
    assert observer.state.valence == result2.valence
    # Поскольку входные данные идентичны, F(t) совпадает
    assert result2.f == pytest.approx(result1.f)
