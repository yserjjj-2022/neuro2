import numpy as np
import pytest

from src.core.energy.calculator import FreeEnergyCalculator
from src.core.energy.models import FreeEnergyResult


@pytest.fixture
def calc() -> FreeEnergyCalculator:
    return FreeEnergyCalculator(dt=0.01, stress_decay=0.99, gamma_base=1.0)


def test_compute_valid(calc: FreeEnergyCalculator) -> None:
    """Формула F(t) на валидных данных."""
    error = np.array([1.0, 2.0])
    precision = np.array([1.0, 1.0])
    result = calc.compute(error, precision, prev_f=0.0, prev_stress=0.0)

    assert isinstance(result, FreeEnergyResult)
    assert result.f == pytest.approx(2.5)  # 0.5 * (1*1 + 1*4) = 2.5
    assert result.gamma == pytest.approx(1.0)


def test_compute_shape_mismatch(calc: FreeEnergyCalculator) -> None:
    """ValueError при несовпадении размерностей."""
    error = np.array([1.0, 2.0])
    precision = np.array([1.0])

    with pytest.raises(ValueError):
        calc.compute(error, precision, prev_f=0.0, prev_stress=0.0)


def test_compute_empty_vs_nonempty_shape_mismatch(calc: FreeEnergyCalculator) -> None:
    """ValueError когда один массив пустой, другой — нет."""
    error = np.array([])
    precision = np.array([0.5, 0.8])

    with pytest.raises(ValueError):
        calc.compute(error, precision, prev_f=0.0, prev_stress=0.0)


def test_compute_empty_arrays(calc: FreeEnergyCalculator) -> None:
    """F(t) = 0.0 для пустых векторов."""
    error = np.array([])
    precision = np.array([])

    result = calc.compute(error, precision, prev_f=0.0, prev_stress=0.0)
    assert result.f == pytest.approx(0.0)


def test_compute_empty_precision(calc: FreeEnergyCalculator) -> None:
    """gamma = gamma_base при пустых векторах."""
    error = np.array([])
    precision = np.array([])

    result = calc.compute(error, precision, prev_f=0.0, prev_stress=0.0)
    assert result.gamma == pytest.approx(1.0)


def test_compute_precision_clip(calc: FreeEnergyCalculator) -> None:
    """Молчаливый clip precision <= 0."""
    error = np.array([-1.0, 0.0])
    precision = np.array([-1.0, 0.0])

    result = calc.compute(error, precision, prev_f=0.0, prev_stress=0.0)
    assert result.gamma > 0


def test_valence_sign(calc: FreeEnergyCalculator) -> None:
    """Проверка знака valence при росте F(t)."""
    result1 = calc.compute(np.array([1.0]), np.array([1.0]), prev_f=0.0, prev_stress=0.0)
    # F(t) выросло с 0.5 до 2.0 -> valence отрицательный
    result2 = calc.compute(np.array([2.0]), np.array([1.0]), prev_f=result1.f, prev_stress=result1.allostatic_stress)
    assert result2.valence < 0


def test_stress_decay(calc: FreeEnergyCalculator) -> None:
    """Монотонное затухание стресса при F(t) = 0."""
    result = calc.compute(np.array([]), np.array([]), prev_f=0.0, prev_stress=10.0)
    assert result.allostatic_stress == pytest.approx(9.9)
