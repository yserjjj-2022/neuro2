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

    result = observer.observe(error, precision)

    assert isinstance(result, FreeEnergyResult)


def test_observer_with_sink(observer: EnergyObserver) -> None:
    """sink вызывается с результатом."""
    log = []
    observer.sink = log.append
    error = np.array([1.0])
    precision = np.array([1.0])

    observer.observe(error, precision)

    assert len(log) == 1
    assert isinstance(log[0], FreeEnergyResult)


def test_observer_maintains_state(observer: EnergyObserver) -> None:
    """Два последовательных observe() корректно передают f(t-1)/stress(t-1)."""
    error1 = np.array([1.0])
    precision1 = np.array([1.0])

    result1 = observer.observe(error1, precision1)

    error2 = np.array([1.0])
    precision2 = np.array([1.0])

    result2 = observer.observe(error2, precision2)

    # Состояние обновлено после второго вызова
    assert observer._prev_f == result2.f
    # Поскольку входные данные идентичны, результат должен совпадать
    assert result2.f == pytest.approx(result1.f)
