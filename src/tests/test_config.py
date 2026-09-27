"""Unit tests for src/config — tunable host parameters (CONSTITUTION §2.2)."""

from __future__ import annotations

import pytest

from src.config import (
    AttractorConfig,
    ColumnParams,
    EnergyConfig,
    HostConfig,
    MemoryConfig,
    SpeechConfig,
)
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

    def test_negative_dt_raises(self) -> None:
        with pytest.raises(ValueError):
            HostConfig(dt=-0.1)

    def test_invalid_precision_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            HostConfig(precision_mode="nope")

    def test_k_exceeds_columns_raises(self) -> None:
        with pytest.raises(ValueError):
            HostConfig(k=5)


class TestMemoryConfig:
    """MemoryConfig: дефолты и валидация (S2)."""

    def test_defaults(self) -> None:
        cfg = MemoryConfig()
        assert cfg.enabled is True
        assert cfg.embedder_mode == "auto"
        assert cfg.embedding_dim == 8
        assert cfg.prior_dim == 4
        assert cfg.recall_limit == 1

    def test_invalid_embedder_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryConfig(embedder_mode="nope")

    def test_invalid_embedding_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryConfig(embedding_dim=0)

    def test_invalid_prior_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryConfig(prior_dim=0)

    def test_invalid_recall_limit_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryConfig(recall_limit=0)

    def test_negative_spike_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryConfig(episode_spike_threshold=-1.0)

    def test_host_config_has_memory(self) -> None:
        assert isinstance(HostConfig().memory, MemoryConfig)


class TestSpeechConfig:
    """SpeechConfig: дефолты и валидация (S3)."""

    def test_defaults(self) -> None:
        cfg = SpeechConfig()
        assert cfg.enabled is False
        assert cfg.llm_mode == "auto"
        assert cfg.default_register == "brief"
        assert cfg.history_turns == 20

    def test_invalid_llm_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            SpeechConfig(llm_mode="nope")

    def test_invalid_register_raises(self) -> None:
        with pytest.raises(ValueError):
            SpeechConfig(default_register="epic")

    def test_invalid_temperature_raises(self) -> None:
        with pytest.raises(ValueError):
            SpeechConfig(temperature=3.0)

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            SpeechConfig(f_threshold=-1.0)

    def test_bad_recall_limit_raises(self) -> None:
        with pytest.raises(ValueError):
            SpeechConfig(recall_limit=0)

    def test_host_config_has_speech(self) -> None:
        assert isinstance(HostConfig().speech, SpeechConfig)
