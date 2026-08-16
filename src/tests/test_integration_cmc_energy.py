"""Integration test: CMC → energy pipeline.

Stable input → e(t) → 0 → F(t) → 0 (free energy falls as columns converge).
Also verifies the wiring pattern: ensemble.active feeds telemetry active_columns.
"""

from __future__ import annotations

import numpy as np

from src.core.cmc.ensemble import CMCEnsemble
from src.core.cmc.models import ColumnConfig
from src.core.energy.calculator import FreeEnergyCalculator


def test_cmc_energy_convergence() -> None:
    """Стабильный вход: e(t) → 0 ⇒ F(t) → 0 (сходимость колонок снижает F)."""
    ensemble = CMCEnsemble(
        columns=[
            ColumnConfig(input_dim=4, state_dim=4, specialization="tone"),
            ColumnConfig(input_dim=4, state_dim=4, specialization="rhythm"),
            ColumnConfig(input_dim=4, state_dim=4, specialization="meaning"),
        ]
    )
    calc = FreeEnergyCalculator()
    u = np.array([1.0, 2.0, 3.0, 4.0])

    prev_f = 0.0
    prev_stress = 0.0
    f_history: list[float] = []

    for _ in range(100):
        out = ensemble.step(u)
        errors = out.errors.ravel()
        precision = np.ones_like(errors)
        result = calc.compute(errors, precision, prev_f, prev_stress)
        prev_f = result.f
        prev_stress = result.allostatic_stress
        f_history.append(result.f)

    # F(t) монотонно убывает к 0 по мере сходимости e(t) → 0
    assert f_history[-1] < f_history[0]
    assert f_history[-1] < 1e-6


def test_cmc_active_feeds_telemetry_pattern() -> None:
    """Паттерн wiring: ensemble.active передаётся как active_columns.

    Первый шаг: все колонки активны (ошибка ≠ 0).
    После сходимости: 0 активных колонок (с порогом — EMA не достигает точно 0).
    """
    ensemble = CMCEnsemble(
        columns=[
            ColumnConfig(input_dim=2, state_dim=2),
            ColumnConfig(input_dim=2, state_dim=2),
        ],
        active_threshold=1e-8,
    )
    u = np.array([1.0, 1.0])

    out = ensemble.step(u)
    assert out.active == 2  # первый шаг — обе колонки активны

    for _ in range(200):
        out = ensemble.step(u)

    assert out.active == 0  # сходимость — ни одна колонка не активна
