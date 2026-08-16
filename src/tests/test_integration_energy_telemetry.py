"""Integration test: energy observer → real telemetry file (JSONL).

Verifies the full pipeline end-to-end — no mocks, real file I/O,
real serialization. This is the ONLY way to catch bugs like
`r.free_energy` vs `r.f` before production.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.host.wiring import build_energy_pipeline


def test_energy_to_telemetry_pipeline(tmp_path: Path) -> None:
    """observer.observe() → реальная запись в JSONL-файл."""
    log_path = tmp_path / "test.jsonl"
    observer = build_energy_pipeline(log_path)

    observer.observe(
        prediction_error=np.array([0.1, 0.2]),
        precision=np.array([1.0, 1.0]),
    )

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1

    event = json.loads(lines[0])
    assert "free_energy" in event
    assert "valence" in event
    assert "allostatic_stress" in event
    assert "active_columns" in event
    assert event["active_columns"] == 0
    assert event["phase"] == "phase1"
    assert event["mode"] == "free"
    # F(t) = 0.5 * (1.0 * 0.01 + 1.0 * 0.04) = 0.025
    assert event["free_energy"] == pytest.approx(0.025)


def test_pipeline_multiple_observations(tmp_path: Path) -> None:
    """Several consecutive observe() calls produce multiple JSONL lines."""
    log_path = tmp_path / "test_multi.jsonl"
    observer = build_energy_pipeline(log_path)

    for _ in range(5):
        observer.observe(
            prediction_error=np.array([0.5]),
            precision=np.array([2.0]),
        )

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 5

    for line in lines:
        event = json.loads(line)
        assert isinstance(event["free_energy"], float)
        assert isinstance(event["valence"], float)
        assert isinstance(event["allostatic_stress"], float)
