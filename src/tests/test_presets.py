"""Tests for named configuration presets (S7-B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import available_presets, load_preset
from src.config.presets import (
    autonomy,
    baseline,
    cooperative,
    dialogue,
    long_horizon,
    stress,
)

_PRESET_FACTORIES = {
    "baseline": baseline,
    "stress": stress,
    "dialogue": dialogue,
    "autonomy": autonomy,
    "long-horizon": long_horizon,
    "cooperative": cooperative,
}


class TestBaseInvariants:
    """Детерминизм базы: synthetic, fake LLM/embedder, фиксированный seed."""

    def test_all_presets_are_deterministic(self) -> None:
        for name, factory in _PRESET_FACTORIES.items():
            assert factory() == factory(), name

    def test_all_presets_synthetic_and_fake(self) -> None:
        for name, factory in _PRESET_FACTORIES.items():
            config = factory()
            assert config.clock_mode == "synthetic", name
            assert config.seed == 0, name
            assert config.memory.embedder_mode == "fake", name
            assert config.speech.llm_mode == "fake", name

    def test_available_presets(self) -> None:
        assert set(available_presets()) == set(_PRESET_FACTORIES)

    def test_dialogue_enables_speech_policy_social(self) -> None:
        config = dialogue()
        assert config.speech.enabled is True
        assert config.policy.enabled is True
        assert config.social.enabled is True

    def test_cooperative_mode(self) -> None:
        assert cooperative().policy.mode == "cooperative"

    def test_autonomy_enabled(self) -> None:
        assert autonomy().autonomy.enabled is True
        assert long_horizon().max_ticks == 1500

    def test_stress_lowers_thresholds(self) -> None:
        config = stress()
        assert config.homeostasis.reflex_threshold == 0.5
        assert config.homeostasis.escape_hatch_ticks == 2


class TestLoadPreset:
    """load_preset: имя + TOML-override с fail-fast."""

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError):
            load_preset("nope")

    def test_without_override_equals_factory(self) -> None:
        assert load_preset("baseline") == baseline()

    def test_override_scalar(self, tmp_path: Path) -> None:
        path = tmp_path / "o.toml"
        path.write_text("[speech]\nf_threshold = 0.25\n")
        config = load_preset("dialogue", override=path)
        assert config.speech.f_threshold == 0.25

    def test_override_nested_section(self, tmp_path: Path) -> None:
        path = tmp_path / "o.toml"
        path.write_text("[policy.preferences]\nalert_deviation = 0.4\n")
        config = load_preset("baseline", override=path)
        assert config.policy.preferences.alert_deviation == 0.4

    def test_override_preserves_other_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "o.toml"
        path.write_text("[speech]\nf_threshold = 0.25\n")
        config = load_preset("dialogue", override=path)
        assert config.speech.default_register == "brief"
        assert config.policy.enabled is True

    def test_unknown_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "o.toml"
        path.write_text("[speech]\nnope = 1\n")
        with pytest.raises(ValueError):
            load_preset("baseline", override=path)

    def test_unknown_top_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "o.toml"
        path.write_text("nope = 1\n")
        with pytest.raises(ValueError):
            load_preset("baseline", override=path)

    def test_invalid_value_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "o.toml"
        path.write_text("[speech]\nf_threshold = -1.0\n")
        with pytest.raises(ValueError):
            load_preset("baseline", override=path)

    def test_section_on_non_dataclass_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "o.toml"
        path.write_text("[seed]\nx = 1\n")
        with pytest.raises(ValueError):
            load_preset("baseline", override=path)


class TestShippedOverrides:
    """Примеры configs/*.toml грузятся поверх базы."""

    def test_dialogue_toml(self) -> None:
        path = Path(__file__).resolve().parents[2] / "configs" / "dialogue.toml"
        config = load_preset("dialogue", override=path)
        assert config.speech.f_threshold == 0.25
        assert config.speech.default_register == "normal"
        assert config.memory.recall_limit == 2

    def test_stress_toml(self) -> None:
        path = Path(__file__).resolve().parents[2] / "configs" / "stress.toml"
        config = load_preset("stress", override=path)
        assert config.homeostasis.escape_hatch_ticks == 1
        assert config.policy.preferences.silent_stress_gain == 0.1
