"""Unit tests for src/config — tunable host parameters (CONSTITUTION §2.2)."""

from __future__ import annotations

import pytest

from src.config import AttractorConfig, ColumnParams, EnergyConfig, HostConfig
from src.core.attractors import TaskAttractor
from src.core.cmc import ColumnConfig
from src.core.energy import FreeEnergyCalculator


class TestEnergyConfig:
    """EnergyConfig → FreeEnergyCalculator."""

    def test_build_returns_calculator(self) -> None:
        calc = EnergyConfig().build()
        assert isinstance(calc, FreeEnergyCalculator)

    def test_values_propagate(self) -> None:
        calc = EnergyConfig(
            stress_leak_per_sec=0.5, valence_tau=0.2, gamma_base=2.0
        ).build()
        assert calc.stress_leak_per_sec == 0.5
        assert calc.valence_tau == 0.2
        assert calc.gamma_base == 2.0


class TestColumnParams:
    """ColumnParams → ColumnConfig под размерности шины."""

    def test_build_returns_config(self) -> None:
        cfg = ColumnParams(specialization="tone", alpha=0.2).build(
            input_dim=12, state_dim=12
        )
        assert isinstance(cfg, ColumnConfig)
        assert cfg.specialization == "tone"
        assert cfg.alpha == 0.2
        assert cfg.input_dim == 12


class TestAttractorConfig:
    """AttractorConfig → TaskAttractor."""

    def test_build_returns_attractor(self) -> None:
        attractor = AttractorConfig().build(n_tasks=3)
        assert isinstance(attractor, TaskAttractor)

    def test_dominance_threshold_propagates(self) -> None:
        """Доминанс-порог из конфига доходит до shell (не хардкод)."""
        attractor = AttractorConfig(dominance_threshold=0.0).build(n_tasks=2)
        assert attractor._dominance_threshold == 0.0


class TestHostConfig:
    """HostConfig: валидация и дефолты."""

    def test_defaults(self) -> None:
        config = HostConfig()
        assert config.k == 2
        assert config.dt == 0.1
        assert config.precision_mode == "variance"
        assert config.clock_mode == "synthetic"
        assert config.gamma_max == 10.0
        assert config.time_scale == 1.0
        assert len(config.columns) == 3

    def test_invalid_clock_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            HostConfig(clock_mode="bogus")

    def test_invalid_precision_window_raises(self) -> None:
        with pytest.raises(ValueError):
            HostConfig(precision_window=0)

    def test_invalid_budgets_raise(self) -> None:
        with pytest.raises(ValueError):
            HostConfig(tick_budget_ms=0.0)
        with pytest.raises(ValueError):
            HostConfig(rss_budget_mb=-1.0)

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            HostConfig(k=0)

    def test_invalid_message_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            HostConfig(message_dim=0)

    def test_negative_dt_raises(self) -> None:
        with pytest.raises(ValueError):
            HostConfig(dt=-0.1)

    def test_invalid_precision_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            HostConfig(precision_mode="nope")

    def test_k_exceeds_columns_raises(self) -> None:
        with pytest.raises(ValueError):
            HostConfig(k=5)
