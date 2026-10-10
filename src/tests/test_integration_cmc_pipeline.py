"""Integration test: CMC → voting → energy pipeline (no I/O).

Telemetry moved to the host loop in S1, so this test verifies the pure
composition via TickOutcome. Full JSONL I/O is covered by test_host_loop.py.
"""

from pathlib import Path

import numpy as np
import pytest

from src.core.cmc.models import ColumnConfig
from src.host.wiring import CMCPipeline, TickOutcome, build_cmc_pipeline


@pytest.fixture()
def pipeline(tmp_path: Path) -> CMCPipeline:
    """Полный конвейер: 3 колонки input_dim=2, k-WTA с k=2."""
    return build_cmc_pipeline(
        columns=[
            ColumnConfig(input_dim=2, state_dim=2, specialization="tone"),
            ColumnConfig(input_dim=2, state_dim=2, specialization="rhythm"),
            ColumnConfig(input_dim=2, state_dim=2, specialization="meaning"),
        ],
        k=2,
        log_path=tmp_path / "test.jsonl",
        active_threshold=1e-8,
    )


def test_pipeline_first_tick(pipeline: CMCPipeline) -> None:
    """Первый тик: активные колонки, F > 0, voting кэширован."""
    u = np.array([1.0, 2.0])
    precision = np.ones(6)  # raveled errors: N_columns * input_dim = 3 * 2

    outcome = pipeline.tick(u, precision, dt=0.01)

    assert isinstance(outcome, TickOutcome)
    assert pipeline.ensemble.active == 3
    assert outcome.result.f > 0.0
    assert outcome.result.gamma == pytest.approx(1.0)

    # voting: результат кэширован в .last
    assert pipeline.voting.last is not None
    assert len(pipeline.voting.last.indices) == 2


def test_pipeline_convergence(pipeline: CMCPipeline) -> None:
    """Стабильный вход: active → 0, F → 0 (ensemble.active — последний step)."""
    u = np.array([1.0, 2.0])
    precision = np.ones(6)

    for _ in range(200):
        outcome = pipeline.tick(u, precision, dt=0.01)

    assert pipeline.ensemble.active == 0
    assert outcome.result.f < 1e-6


def test_pipeline_shape_mismatch_precision(pipeline: CMCPipeline) -> None:
    """Несовпадение precision и raveled errors → ValueError из tick()."""
    u = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        pipeline.tick(u, np.ones(5), dt=0.01)  # ожидается 6

    with pytest.raises(ValueError):
        pipeline.tick(u, np.ones(7), dt=0.01)


def test_pipeline_shape_mismatch_u(pipeline: CMCPipeline) -> None:
    """Несовпадение u и input_dim → ValueError из tick()."""
    with pytest.raises(ValueError):
        pipeline.tick(np.array([1.0, 2.0, 3.0]), np.ones(6), dt=0.01)


def test_pipeline_active_tags(pipeline: CMCPipeline) -> None:
    """active_tags заполняются по карте сегментов шины."""
    from src.host.sources import BusSegment

    segments = (
        BusSegment(name="a", offset=0, dim=1, period=1),
        BusSegment(name="b", offset=1, dim=1, period=1),
    )
    u = np.array([1.0, 2.0])
    outcome = pipeline.tick(u, np.ones(6), dt=0.01, segments=segments)
    # Первый тик: ошибка по обоим каналам > порога
    assert outcome.active_tags == ("a", "b")


def test_pipeline_channel_contrib(pipeline: CMCPipeline) -> None:
    """channel_contrib отражает вклад каналов и реагирует на importance."""
    from src.host.sources import BusSegment

    segments = (
        BusSegment(name="a", offset=0, dim=1, period=1),
        BusSegment(name="b", offset=1, dim=1, period=1),
    )
    u = np.array([1.0, 2.0])
    outcome = pipeline.tick(u, np.ones(6), dt=0.01, segments=segments)

    tags = [tag for tag, _ in outcome.channel_contrib]
    assert tags == ["a", "b"]
    assert all(value >= 0.0 for _, value in outcome.channel_contrib)

    # Канал b (вход 2.0) даёт больший вклад, чем a (вход 1.0)
    contrib = dict(outcome.channel_contrib)
    assert contrib["b"] > contrib["a"]

    # importance усиливает вклад: удвоение веса a поднимает его вклад
    weighted = pipeline.tick(
        u, np.ones(6), dt=0.01, segments=segments, importance=np.array([8.0, 1.0])
    )
    assert dict(weighted.channel_contrib)["a"] > contrib["a"]


def test_pipeline_channel_contrib_empty_without_segments(
    pipeline: CMCPipeline,
) -> None:
    """Без карты сегментов channel_contrib пуст."""
    outcome = pipeline.tick(np.array([1.0, 2.0]), np.ones(6), dt=0.01)
    assert outcome.channel_contrib == ()
