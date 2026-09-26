import numpy as np
import pytest

from src.core.energy.calculator import FreeEnergyCalculator
from src.core.energy.models import EnergyState, FreeEnergyResult


@pytest.fixture
def calc() -> FreeEnergyCalculator:
    return FreeEnergyCalculator(
        stress_leak_per_sec=1.0, valence_tau=0.1, gamma_base=1.0
    )


def test_compute_valid(calc: FreeEnergyCalculator) -> None:
    """Формула F(t) на валидных данных."""
    error = np.array([1.0, 2.0])
    precision = np.array([1.0, 1.0])
    result = calc.compute(error, precision, EnergyState(), dt=0.01)

    assert isinstance(result, FreeEnergyResult)
    assert result.f == pytest.approx(2.5)  # 0.5 * (1*1 + 1*4) = 2.5
    assert result.gamma == pytest.approx(1.0)


def test_compute_shape_mismatch(calc: FreeEnergyCalculator) -> None:
    """ValueError при несовпадении размерностей."""
    error = np.array([1.0, 2.0])
    precision = np.array([1.0])

    with pytest.raises(ValueError):
        calc.compute(error, precision, EnergyState(), dt=0.01)


def test_compute_invalid_dt(calc: FreeEnergyCalculator) -> None:
    """ValueError при dt <= 0 (единая временная база S1)."""
    error = np.array([1.0])
    precision = np.array([1.0])

    with pytest.raises(ValueError):
        calc.compute(error, precision, EnergyState(), dt=0.0)
    with pytest.raises(ValueError):
        calc.compute(error, precision, EnergyState(), dt=-0.1)


def test_compute_empty_vs_nonempty_shape_mismatch(calc: FreeEnergyCalculator) -> None:
    """ValueError когда один массив пустой, другой — нет."""
    error = np.array([])
    precision = np.array([0.5, 0.8])

    with pytest.raises(ValueError):
        calc.compute(error, precision, EnergyState(), dt=0.01)


def test_compute_empty_arrays(calc: FreeEnergyCalculator) -> None:
    """F(t) = 0.0 для пустых векторов."""
    error = np.array([])
    precision = np.array([])

    result = calc.compute(error, precision, EnergyState(), dt=0.01)
    assert result.f == pytest.approx(0.0)


def test_compute_empty_precision(calc: FreeEnergyCalculator) -> None:
    """gamma = gamma_base при пустых векторах."""
    error = np.array([])
    precision = np.array([])

    result = calc.compute(error, precision, EnergyState(), dt=0.01)
    assert result.gamma == pytest.approx(1.0)


def test_compute_precision_clip(calc: FreeEnergyCalculator) -> None:
    """Молчаливый clip precision <= 0."""
    error = np.array([-1.0, 0.0])
    precision = np.array([-1.0, 0.0])

    result = calc.compute(error, precision, EnergyState(), dt=0.01)
    assert result.gamma > 0


def test_valence_sign(calc: FreeEnergyCalculator) -> None:
    """Проверка знака valence при росте F(t)."""
    result1 = calc.compute(np.array([1.0]), np.array([1.0]), EnergyState(), dt=0.01)
    # F(t) выросло с 0.5 до 2.0 -> valence отрицательный
    state = EnergyState(
        f=result1.f, stress=result1.allostatic_stress, valence=result1.valence
    )
    result2 = calc.compute(np.array([2.0]), np.array([1.0]), state, dt=0.01)
    assert result2.valence < 0


def test_stress_leak(calc: FreeEnergyCalculator) -> None:
    """Экспоненциальная утечка стресса при F(t) = 0."""
    result = calc.compute(np.array([]), np.array([]), EnergyState(stress=10.0), dt=0.01)
    # 10 * exp(-1.0 * 0.01) = 10 * 0.99005 = 9.9005
    assert result.allostatic_stress == pytest.approx(9.9005, rel=1e-4)


def test_valence_smoothing_reduces_jitter(calc: FreeEnergyCalculator) -> None:
    """Сглаженная valence не превышает raw по модулю (EMA-фильтр)."""
    # Скачок F: raw valence огромна, сглаженная — доля от неё
    state = EnergyState(f=0.0, stress=0.0, valence=0.0)
    result = calc.compute(np.array([10.0]), np.array([1.0]), state, dt=0.01)
    raw = -(result.f - 0.0) / 0.01  # -5000
    assert abs(result.valence) < abs(raw)


def test_dt_scaling_stress(calc: FreeEnergyCalculator) -> None:
    """Накопление стресса зависит от dt (секунды, не тики)."""
    error = np.array([1.0])
    precision = np.array([1.0])
    r_small = calc.compute(error, precision, EnergyState(), dt=0.001)
    r_large = calc.compute(error, precision, EnergyState(), dt=0.01)
    # f одинаков, но f*dt различается в 10 раз
    assert r_large.allostatic_stress == pytest.approx(
        r_small.allostatic_stress * 10, rel=1e-6
    )
