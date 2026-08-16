"""Integration test: CMC → voting pipeline.

Column activities (‖e‖² from CMCEnsemble) → scores → kwta winners.
Stable input → errors → 0 → zero activities → top-k on zero scores
resolves via ties policy (smaller index wins).
"""

from __future__ import annotations

import numpy as np

from src.core.cmc.ensemble import CMCEnsemble
from src.core.cmc.models import ColumnConfig
from src.core.voting.kwta import kwta


def test_voting_cmc_winners_are_highest_error_columns() -> None:
    """Победители — колонки с наибольшей ошибкой предсказания (активностью)."""
    ensemble = CMCEnsemble(
        columns=[
            ColumnConfig(input_dim=3, state_dim=3, specialization="tone"),
            ColumnConfig(input_dim=3, state_dim=3, specialization="rhythm"),
            ColumnConfig(input_dim=3, state_dim=3, specialization="meaning"),
        ]
    )
    # Первый шаг: каждая колонка получает свою ошибку
    out = ensemble.step(u=np.array([1.0, 2.0, 3.0]))

    # Активности колонок = ‖e‖² по строкам errors (цепочка cmc → voting)
    activities = np.sum(out.errors**2, axis=1)
    result = kwta(activities, k=2)

    # Победители — 2 колонки с максимальной активностью
    assert len(result.indices) == 2
    assert result.mask.sum() == 2
    top2 = np.argsort(-activities)[:2]
    np.testing.assert_array_equal(result.indices, top2)


def test_voting_cmc_convergence_ties() -> None:
    """Сходимость → нулевые активности → top-k по нулям → меньшие индексы."""
    ensemble = CMCEnsemble(
        columns=[
            ColumnConfig(input_dim=2, state_dim=2),
            ColumnConfig(input_dim=2, state_dim=2),
        ]
    )
    u = np.array([1.0, 1.0])
    ensemble.step(u)

    for _ in range(200):
        out = ensemble.step(u)

    activities = np.sum(out.errors**2, axis=1)
    # Все активности ≈ 0 (сходимость EMA)
    np.testing.assert_allclose(activities, 0.0, atol=1e-8)

    # top-k по нулевым scores: ties policy → меньшие индексы
    result = kwta(activities, k=1)
    assert result.indices.tolist() == [0]
