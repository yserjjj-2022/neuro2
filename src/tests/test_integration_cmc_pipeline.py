"""Integration test: CMC → voting → energy → telemetry pipeline.

Verifies the full per-tick pipeline end-to-end — no mocks, real file I/O,
real serialization. This is the ONLY way to catch wiring bugs like
stale active_columns cache or shape mismatch between precision and
raveled errors before production.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.cmc.models import ColumnConfig
from src.host.wiring import CMCPipeline, build_cmc_pipeline


@pytest.fixture()
def pipeline(tmp_path: Path) -> CMCPipeline:
    """Полный конвейер: 3 колонки input_dim=2, k-WTA с k=2.

    active_threshold=1e-8: EMA-сходимость не достигает точно нуля в float64
    (остаток ~6e-10), порог позволяет проверять «колонка сошлась → не активна».
    """
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


def test_pipeline_first_tick(pipeline: CMCPipeline, tmp_path: Path) -> None:
    """Первый тик: активные колонки, F > 0, active_columns из cmc в JSONL."""
    u = np.array([1.0, 2.0])
    precision = np.ones(6)  # raveled errors: N_columns * input_dim = 3 * 2

    result = pipeline.tick(u, precision)

    # Первый шаг из нулей: все колонки активны, F > 0
    assert pipeline.ensemble.active == 3
    assert result.f > 0.0

    # voting: результат кэширован в .last для будущих аттракторов
    assert pipeline.voting.last is not None
    assert len(pipeline.voting.last.indices) == 2

    # Телеметрия: active_columns из cmc (не устаревший дефолт 0)
    lines = (tmp_path / "test.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["active_columns"] == 3
    assert event["free_energy"] == pytest.approx(result.f)
    assert event["phase"] == "phase1"
    assert event["mode"] == "free"


def test_pipeline_convergence(pipeline: CMCPipeline, tmp_path: Path) -> None:
    """Стабильный вход: active → 0, F → 0, active_columns следует за тиками.

    Проверяет, что ensemble.active — результат ПОСЛЕДНЕГО step(), а не
    устаревшее значение из кэша первого тика.
    """
    u = np.array([1.0, 2.0])
    precision = np.ones(6)

    for _ in range(200):
        result = pipeline.tick(u, precision)

    assert pipeline.ensemble.active == 0
    assert result.f < 1e-6

    lines = (tmp_path / "test.jsonl").read_text().strip().split("\n")
    assert len(lines) == 200

    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    assert first["active_columns"] == 3  # первый тик — все активны
    assert last["active_columns"] == 0  # последний тик — все сошлись
    assert last["free_energy"] < 1e-6


def test_pipeline_shape_mismatch_precision(pipeline: CMCPipeline) -> None:
    """Несовпадение precision и raveled errors → ValueError из tick() (fail-fast)."""
    u = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        pipeline.tick(u, np.ones(5))  # ожидается 6

    with pytest.raises(ValueError):
        pipeline.tick(u, np.ones(7))


def test_pipeline_shape_mismatch_u(pipeline: CMCPipeline) -> None:
    """Несовпадение u и input_dim → ValueError из tick() (fail-fast)."""
    with pytest.raises(ValueError):
        pipeline.tick(np.array([1.0, 2.0, 3.0]), np.ones(6))  # input_dim=2
