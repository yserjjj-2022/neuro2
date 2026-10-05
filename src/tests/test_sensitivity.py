"""Tests for the sensitivity harness Core + Shell (S7-A)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.host.sensitivity import (
    SensitivityCase,
    SensitivityRunner,
    _apply_knob,
    _base_config,
    build_sensitivity_matrix,
    check_direction,
)


class TestCheckDirection:
    """Инварианты направления/границ (Core)."""

    def test_nondecreasing_ok(self) -> None:
        assert check_direction([1.0, 1.0, 2.0], direction="nondecreasing")

    def test_nondecreasing_fail(self) -> None:
        assert not check_direction([2.0, 1.0], direction="nondecreasing")

    def test_nonincreasing_ok(self) -> None:
        assert check_direction([2.0, 2.0, 1.0], direction="nonincreasing")

    def test_nonincreasing_fail(self) -> None:
        assert not check_direction([1.0, 2.0], direction="nonincreasing")

    def test_bounded(self) -> None:
        assert check_direction([0.0, 0.5, 1.0], direction="bounded")
        assert not check_direction([0.0, 1.5], direction="bounded")

    def test_non_finite_fails(self) -> None:
        assert not check_direction([0.0, float("nan")], direction="bounded")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            check_direction([], direction="bounded")

    def test_unknown_direction_raises(self) -> None:
        with pytest.raises(ValueError):
            check_direction([0.0], direction="sideways")


class TestSensitivityCase:
    """Валидация ячейки матрицы (Core)."""

    def test_empty_values_raise(self) -> None:
        with pytest.raises(ValueError):
            SensitivityCase("x", (), "talk_rate", "bounded")

    def test_bad_direction_raise(self) -> None:
        with pytest.raises(ValueError):
            SensitivityCase("x", (1.0,), "talk_rate", "sideways")

    def test_bad_metric_raise(self) -> None:
        with pytest.raises(ValueError):
            SensitivityCase("x", (1.0,), "nope", "bounded")

    def test_matrix_valid(self) -> None:
        matrix = build_sensitivity_matrix()
        assert matrix
        assert all(isinstance(case, SensitivityCase) for case in matrix)


class TestApplyKnob:
    """Возмущение ручки доходит до конфига (fail-fast)."""

    def test_preference_knob(self, tmp_path: Path) -> None:
        config = _base_config(tmp_path, seed=0, ticks=5)
        updated = _apply_knob(config, "silent_stress_gain", 2.0)
        assert updated.policy.preferences.silent_stress_gain == 2.0
        # База не мутирована (frozen).
        assert config.policy.preferences.silent_stress_gain != 2.0

    def test_unknown_knob_raises(self, tmp_path: Path) -> None:
        config = _base_config(tmp_path, seed=0, ticks=5)
        with pytest.raises(ValueError):
            _apply_knob(config, "nope", 1.0)

    def test_all_matrix_knobs_applicable(self, tmp_path: Path) -> None:
        config = _base_config(tmp_path, seed=0, ticks=5)
        for case in build_sensitivity_matrix():
            _apply_knob(config, case.knob, case.values[0])


class TestSensitivityRunner:
    """Прогон матрицы поверх HostLoop (Shell)."""

    def test_run_all_directions_hold(self, tmp_path: Path) -> None:
        runner = SensitivityRunner(ticks=80, workdir=tmp_path)
        results = runner.run_all(seed=0)
        assert results
        for result in results:
            assert result.passed, (
                f"{result.case.knob}->{result.case.metric} "
                f"({result.case.direction}) got {result.metric_values}"
            )

    def test_determinism_same_seed(self, tmp_path: Path) -> None:
        case = build_sensitivity_matrix()[0]
        a = SensitivityRunner(ticks=40, workdir=tmp_path / "a").run_case(case, seed=7)
        b = SensitivityRunner(ticks=40, workdir=tmp_path / "b").run_case(case, seed=7)
        assert a == b

    def test_rates_bounded(self, tmp_path: Path) -> None:
        runner = SensitivityRunner(ticks=40, workdir=tmp_path)
        for case in build_sensitivity_matrix():
            values = runner.run_case(case, seed=1)
            assert all(0.0 <= v <= 1.0 for v in values), (case, values)

    def test_bad_ticks_raise(self) -> None:
        with pytest.raises(ValueError):
            SensitivityRunner(ticks=0)
