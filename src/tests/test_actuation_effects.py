"""Unit tests for actuation effects and contract (S8 stage 3, ADR-0012).

Covers ``Effect`` validation, the actuation contract enums, the conservative
default of ``classify_reversible`` (untrusted source never grants autonomy),
trust grading, and additive enrichment (an option without guard/effect still
scores as before — stage 1 compatibility).
"""

from __future__ import annotations

import pytest

from src.core.actuation import (
    Actuation,
    ActuationKind,
    ActuationPreferences,
    ActuationResult,
    ActuationStatus,
    Effect,
    Fact,
    Guard,
    Option,
    OptionContext,
    OptionSource,
    OptionWindow,
    ToolAnnotations,
    classify_reversible,
    score_option,
    select_option,
)


def _fact(name: str = "network_available") -> Fact:
    """Собрать факт для теста."""
    return Fact(name=name)


class TestEffect:
    """Tests for Effect — symbolic fact-delta."""

    def test_default_value_is_one(self) -> None:
        """Дефолтное целевое значение — 1 (факт истинен)."""
        assert Effect(_fact()).value == 1.0

    def test_holds_fact_and_value(self) -> None:
        """Эффект несёт факт и целевое значение."""
        effect = Effect(_fact("x"), value=0.7)
        assert effect.fact.name == "x"
        assert effect.value == 0.7

    def test_value_above_one_rejected(self) -> None:
        """value > 1 → ValueError."""
        with pytest.raises(ValueError, match="effect value must be in"):
            Effect(_fact(), value=1.5)

    def test_value_below_zero_rejected(self) -> None:
        """value < 0 → ValueError."""
        with pytest.raises(ValueError, match="effect value must be in"):
            Effect(_fact(), value=-0.1)


class TestActuationContract:
    """Tests for the unified actuation contract (kind/goal/payload/status)."""

    def test_kinds(self) -> None:
        """Два вида актуации: речь и вызов тула."""
        assert {k.value for k in ActuationKind} == {"speak", "invoke_tool"}

    def test_statuses(self) -> None:
        """Четыре статуса (идиома ROS Action Server)."""
        assert {s.value for s in ActuationStatus} == {
            "running",
            "success",
            "failure",
            "preempted",
        }

    def test_actuation_carries_goal(self) -> None:
        """Актуация несёт вид, цель и payload."""
        act = Actuation(ActuationKind.INVOKE_TOOL, "tool:get_weather", "get_weather")
        assert act.kind is ActuationKind.INVOKE_TOOL
        assert act.goal == "tool:get_weather"
        assert act.payload == "get_weather"

    def test_empty_goal_rejected(self) -> None:
        """Пустая цель → ValueError."""
        with pytest.raises(ValueError, match="actuation goal must not be empty"):
            Actuation(ActuationKind.SPEAK, "")

    def test_result_default_data_empty(self) -> None:
        """Результат по умолчанию без данных."""
        result = ActuationResult(ActuationStatus.SUCCESS)
        assert result.status is ActuationStatus.SUCCESS
        assert result.data == ()

    def test_result_carries_data(self) -> None:
        """Результат несёт данные для шины."""
        result = ActuationResult(ActuationStatus.SUCCESS, (0.1, 0.2))
        assert result.data == (0.1, 0.2)


class TestClassifyReversible:
    """Tests for classify_reversible — irreversibility as a stance."""

    def test_untrusted_always_false(self) -> None:
        """Недоверенный источник → необратимо, даже при read_only."""
        read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False)
        assert classify_reversible(read_only, trusted=False) is False

    def test_trusted_read_only_reversible(self) -> None:
        """Доверенный read-only тул → обратим."""
        ann = ToolAnnotations(read_only_hint=True, destructive_hint=False)
        assert classify_reversible(ann, trusted=True) is True

    def test_trusted_destructive_not_reversible(self) -> None:
        """Доверенный деструктивный тул → необратим."""
        ann = ToolAnnotations(read_only_hint=True, destructive_hint=True)
        assert classify_reversible(ann, trusted=True) is False

    def test_default_annotations_conservative(self) -> None:
        """Дефолтные аннотации (destructive=True) → необратимо даже доверенно."""
        assert classify_reversible(ToolAnnotations(), trusted=True) is False

    def test_deterministic(self) -> None:
        """Одинаковый вход → одинаковый выход."""
        ann = ToolAnnotations(read_only_hint=True, destructive_hint=False)
        assert classify_reversible(ann, trusted=True) == classify_reversible(
            ann, trusted=True
        )


class TestAdditiveEnrichment:
    """Tests for additive Option.guard/effect — stage 1 compatibility."""

    def test_option_defaults_none(self) -> None:
        """Дефолт: guard/effect = None (совместимость с этапами 1–2)."""
        option = Option("tool:x", OptionSource.TOOL)
        assert option.guard is None
        assert option.effect is None

    def test_option_accepts_guard_effect(self) -> None:
        """Опция принимает guard и effect (аддитивное обогащение)."""
        guard = Guard(_fact(), threshold=0.5)
        effect = Effect(_fact(), value=1.0)
        option = Option("tool:x", OptionSource.TOOL, guard=guard, effect=effect)
        assert option.guard is guard
        assert option.effect is effect

    def test_option_without_enrichment_scores_as_before(self) -> None:
        """Опция без обогащения оценивается как эпистемическая (этап 1)."""
        option = Option("tool:x", OptionSource.TOOL)
        pragmatic, epistemic, _ = score_option(
            option, OptionContext(uncertainty=0.8), ActuationPreferences()
        )
        assert pragmatic == 0.0
        assert epistemic == pytest.approx(0.8)

    def test_select_option_still_works(self) -> None:
        """select_option работает с опциями без обогащения (совместимость)."""
        window = OptionWindow((Option("tool:x", OptionSource.TOOL, reversible=True),))
        trace = select_option(window, OptionContext(uncertainty=0.5))
        assert trace.chosen is window.options[0]
