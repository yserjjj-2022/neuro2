"""Unit tests for CMCEnsemble — Imperative Shell.

Shell owns column states; tested with small in-memory ensembles.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.cmc.ensemble import CMCEnsemble
from src.core.cmc.models import ColumnConfig


@pytest.fixture()
def ensemble() -> CMCEnsemble:
    return CMCEnsemble(
        columns=[
            ColumnConfig(input_dim=3, state_dim=3, specialization="tone"),
            ColumnConfig(input_dim=3, state_dim=3, specialization="rhythm"),
            ColumnConfig(input_dim=3, state_dim=3, specialization="meaning"),
        ]
    )


def test_ensemble_init_empty() -> None:
    """Пустой список колонок → ValueError."""
    with pytest.raises(ValueError):
        CMCEnsemble(columns=[])


def test_ensemble_init_mismatched_dims() -> None:
    """Разные input_dim/state_dim у колонок → ValueError."""
    with pytest.raises(ValueError):
        CMCEnsemble(
            columns=[
                ColumnConfig(input_dim=3, state_dim=3),
                ColumnConfig(input_dim=5, state_dim=5),
            ]
        )


def test_ensemble_step_aggregation(ensemble: CMCEnsemble) -> None:
    """N колонок → errors shape (N, input_dim), states shape (N, state_dim)."""
    u = np.array([1.0, 2.0, 3.0])

    out = ensemble.step(u)

    assert out.errors.shape == (3, 3)
    assert out.states.shape == (3, 3)


def test_ensemble_step_shape_mismatch(ensemble: CMCEnsemble) -> None:
    """u.shape != (input_dim,) → ValueError."""
    u = np.array([1.0, 2.0])  # (2,) vs (3,)

    with pytest.raises(ValueError):
        ensemble.step(u)


def test_ensemble_active(ensemble: CMCEnsemble) -> None:
    """Колонки с ненулевой ошибкой → active считает правильно.

    Note: EMA-сходимость приближается к 0, но не достигает точно 0 в float64
    (остаток ~1e-10). Поэтому для проверки "сходилась → не активна" нужен
    ненулевой порог. Дефолт 0.0 = "любая ненулевая ошибка активна" — корректен
    для первого шага, но не для сходимости.
    """
    u = np.array([1.0, 2.0, 3.0])

    out = ensemble.step(u)

    # Первый шаг из нулей: все 3 колонки получили ошибку u - α·u ≠ 0
    assert out.active == 3

    # Стабильный вход 200 шагов с порогом 1e-8 → ошибки < порога → active → 0
    converged = CMCEnsemble(
        columns=[
            ColumnConfig(input_dim=3, state_dim=3, specialization="tone"),
            ColumnConfig(input_dim=3, state_dim=3, specialization="rhythm"),
            ColumnConfig(input_dim=3, state_dim=3, specialization="meaning"),
        ],
        active_threshold=1e-8,
    )
    for _ in range(200):
        out = converged.step(u)

    assert out.active == 0


def test_ensemble_reset(ensemble: CMCEnsemble) -> None:
    """После reset все состояния нулевые, active == 0."""
    u = np.array([1.0, 2.0, 3.0])
    ensemble.step(u)

    ensemble.reset()

    assert ensemble.active == 0
    # Следующий step из нулей снова даёт активные колонки
    out = ensemble.step(u)
    assert out.active == 3


def test_ensemble_active_before_step() -> None:
    """active property == 0 до первого step()."""
    ensemble = CMCEnsemble(columns=[ColumnConfig(input_dim=3, state_dim=3)])
    assert ensemble.active == 0


def test_ensemble_default_threshold() -> None:
    """active_threshold по умолчанию = 1e-8 (EMA never converges to exact 0.0)."""
    ensemble = CMCEnsemble(columns=[ColumnConfig(input_dim=3, state_dim=3)])
    assert ensemble.active_threshold == pytest.approx(1e-8)


def test_ensemble_active_threshold() -> None:
    """active_threshold > 0: только колонки с ||e||² > threshold активны."""
    ensemble = CMCEnsemble(
        columns=[ColumnConfig(input_dim=1, state_dim=1, alpha=0.5)],
        active_threshold=0.5,
    )
    u = np.array([1.0])
    out = ensemble.step(u)
    # e = u - x = 1.0 - 0.5*1.0 = 0.5; ||e||² = 0.25 < 0.5 → не активна
    assert out.active == 0

    ensemble.reset()
    u = np.array([2.0])
    out = ensemble.step(u)
    # e = 2.0 - 1.0 = 1.0; ||e||² = 1.0 > 0.5 → активна
    assert out.active == 1
