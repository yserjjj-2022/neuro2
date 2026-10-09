"""Unit tests for actuation facts and conditions (S8 stage 2, ADR-0012).

Covers totality (an unknown fact degrades to its default, never crashes),
clipping (a value outside [0, 1] is clamped), guard threshold boundaries,
monotonicity of regularity cost, empty input, determinism, the fact registry,
and input validation.
"""

from __future__ import annotations

import pytest

from src.core.actuation import (
    DEFAULT_FACT_VALUE,
    FACTS,
    NETWORK_AVAILABLE,
    TOPIC_BOUND,
    Fact,
    Guard,
    Regularity,
    evaluate_fact,
    guard_holds,
    regularity_cost,
)


def _fact(name: str = "probe", default: float = DEFAULT_FACT_VALUE) -> Fact:
    """Собрать факт для теста."""
    return Fact(name=name, default=default)


class TestEvaluateFact:
    """Tests for evaluate_fact — total read of a fact from a world snapshot."""

    def test_reads_value_from_state(self) -> None:
        """Известный факт читается из снимка мира."""
        assert (
            evaluate_fact(_fact("network_available"), {"network_available": 0.7}) == 0.7
        )

    def test_unknown_fact_degrades_to_default(self) -> None:
        """Неизвестный факт → fact.default (не падение)."""
        assert evaluate_fact(_fact("missing", default=0.25), {}) == 0.25

    def test_unknown_fact_defaults_to_neutral(self) -> None:
        """Дефолт без обогащения — нейтральное значение."""
        assert evaluate_fact(_fact("missing"), {}) == DEFAULT_FACT_VALUE

    def test_clips_above_one(self) -> None:
        """Значение > 1 зажимается к 1."""
        assert evaluate_fact(_fact("x"), {"x": 3.0}) == 1.0

    def test_clips_below_zero(self) -> None:
        """Значение < 0 зажимается к 0."""
        assert evaluate_fact(_fact("x"), {"x": -2.0}) == 0.0

    def test_total_over_arbitrary_names(self) -> None:
        """Тотальность: определён для любого имени факта."""
        for name in ("a", "b", "some/deep:name", "unknown_42"):
            assert 0.0 <= evaluate_fact(_fact(name), {"a": 0.5}) <= 1.0

    def test_deterministic(self) -> None:
        """Одинаковый вход → одинаковый выход."""
        fact = _fact("x", default=0.3)
        assert evaluate_fact(fact, {}) == evaluate_fact(fact, {})


class TestGuardHolds:
    """Tests for guard_holds — hard condition (threshold)."""

    def test_holds_at_threshold(self) -> None:
        """fact == threshold → проходит (>=)."""
        guard = Guard(_fact("x"), threshold=0.5)
        assert guard_holds(guard, {"x": 0.5}) is True

    def test_holds_above_threshold(self) -> None:
        """fact > threshold → проходит."""
        guard = Guard(_fact("x"), threshold=0.5)
        assert guard_holds(guard, {"x": 0.9}) is True

    def test_fails_below_threshold(self) -> None:
        """fact < threshold → не проходит."""
        guard = Guard(_fact("x"), threshold=0.5)
        assert guard_holds(guard, {"x": 0.4}) is False

    def test_unknown_fact_uses_default(self) -> None:
        """Неизвестный факт → дефолт; дефолт ниже порога → не проходит."""
        guard = Guard(_fact("missing", default=0.0), threshold=0.5)
        assert guard_holds(guard, {}) is False

    def test_zero_threshold_always_holds(self) -> None:
        """Порог 0 проходит всегда (значение >= 0)."""
        guard = Guard(_fact("x"), threshold=0.0)
        assert guard_holds(guard, {"x": 0.0}) is True


class TestRegularityCost:
    """Tests for regularity_cost — soft conditions become cost."""

    def test_empty_is_zero(self) -> None:
        """Пустой список → 0.0 (дефолт без обогащения)."""
        assert regularity_cost([], {}) == 0.0

    def test_weighted_sum(self) -> None:
        """Стоимость = Σ weight·fact."""
        regs = [Regularity(_fact("a"), weight=2.0), Regularity(_fact("b"), weight=0.5)]
        assert regularity_cost(regs, {"a": 0.5, "b": 1.0}) == pytest.approx(1.5)

    def test_unknown_facts_use_defaults(self) -> None:
        """Неизвестные факты → дефолты, не падение."""
        regs = [Regularity(_fact("missing", default=0.2), weight=1.0)]
        assert regularity_cost(regs, {}) == pytest.approx(0.2)

    def test_monotone_in_fact_value(self) -> None:
        """Монотонность: рост факта не уменьшает стоимость (вес ≥ 0)."""
        regs = [Regularity(_fact("x"), weight=1.0)]
        low = regularity_cost(regs, {"x": 0.2})
        high = regularity_cost(regs, {"x": 0.8})
        assert high >= low

    def test_zero_weight_ignores_fact(self) -> None:
        """Нулевой вес → факт не влияет на стоимость."""
        regs = [Regularity(_fact("x"), weight=0.0)]
        assert regularity_cost(regs, {"x": 1.0}) == 0.0

    def test_deterministic(self) -> None:
        """Одинаковый вход → одинаковый выход."""
        regs = [Regularity(_fact("x"), weight=1.5)]
        state = {"x": 0.4}
        assert regularity_cost(regs, state) == regularity_cost(regs, state)


class TestFactsRegistry:
    """Tests for the FACTS registry — single source of names."""

    def test_registry_keys_match_names(self) -> None:
        """Ключ реестра = имя факта (Core и Shell не расходятся)."""
        for key, fact in FACTS.items():
            assert key == fact.name

    def test_registry_nonempty(self) -> None:
        """Реестр содержит начальный набор фактов."""
        assert NETWORK_AVAILABLE.name in FACTS
        assert TOPIC_BOUND.name in FACTS

    def test_registry_values_are_facts(self) -> None:
        """Все значения реестра — Fact."""
        assert all(isinstance(fact, Fact) for fact in FACTS.values())


class TestValidation:
    """Tests for input validation — ValueError on bad data."""

    def test_fact_empty_name(self) -> None:
        """Пустое имя факта → ValueError."""
        with pytest.raises(ValueError, match="fact name must not be empty"):
            Fact(name="")

    def test_fact_default_out_of_range(self) -> None:
        """default вне [0, 1] → ValueError."""
        with pytest.raises(ValueError, match="fact default must be in"):
            Fact(name="x", default=1.5)

    def test_guard_threshold_out_of_range(self) -> None:
        """threshold вне [0, 1] → ValueError."""
        with pytest.raises(ValueError, match="threshold must be in"):
            Guard(_fact("x"), threshold=-0.1)

    def test_regularity_negative_weight(self) -> None:
        """weight < 0 → ValueError."""
        with pytest.raises(ValueError, match="weight must be >= 0"):
            Regularity(_fact("x"), weight=-1.0)
