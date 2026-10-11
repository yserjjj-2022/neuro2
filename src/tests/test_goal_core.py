"""Unit tests for the goal-loop Core functions (S8 pass 2, ADR-0014/0015).

Covers ``PreferredOutcome`` validation, ``unsatisfied_outcomes``,
``select_goal``, ``goal_reached``, ``effect_gain`` (fraction-of-gap-closed
formula, ``effect.value`` as predicted final value), the additive pragmatic
score in ``score_option`` and the unified-scalar ranking in ``backward_chain``.
"""

from __future__ import annotations

import pytest

from src.core.actuation import (
    ActuationPreferences,
    Effect,
    Fact,
    Goal,
    Option,
    OptionContext,
    OptionSource,
    PreferredOutcome,
    backward_chain,
    effect_gain,
    goal_reached,
    score_option,
    select_goal,
    unsatisfied_outcomes,
)

_SERVICED = Fact("serviced")
_OTHER = Fact("other")


def _option(
    option_id: str,
    *,
    effect: Fact | None = None,
    effect_value: float = 1.0,
) -> Option:
    """Собрать опцию с (возможным) эффектом для теста."""
    return Option(
        option_id,
        OptionSource.TOOL,
        effect=None if effect is None else Effect(effect, value=effect_value),
    )


class TestPreferredOutcome:
    """Валидация PreferredOutcome."""

    def test_defaults(self) -> None:
        """Дефолты: target=1, priority=1."""
        outcome = PreferredOutcome(_SERVICED)
        assert outcome.target == 1.0
        assert outcome.priority == 1.0

    def test_target_above_one_rejected(self) -> None:
        """target > 1 → ValueError."""
        with pytest.raises(ValueError, match="target must be in"):
            PreferredOutcome(_SERVICED, target=1.5)

    def test_negative_priority_rejected(self) -> None:
        """priority < 0 → ValueError."""
        with pytest.raises(ValueError, match="priority must be >= 0"):
            PreferredOutcome(_SERVICED, priority=-0.1)


class TestUnsatisfiedOutcomes:
    """Отбор неудовлетворённых исходов."""

    def test_empty_outcomes(self) -> None:
        """Пустой вход → ()."""
        assert unsatisfied_outcomes((), {}) == ()

    def test_filters_satisfied(self) -> None:
        """Удовлетворённый исход не попадает в результат."""
        outcomes = (PreferredOutcome(_SERVICED, target=1.0),)
        assert unsatisfied_outcomes(outcomes, {"serviced": 1.0}) == ()

    def test_unknown_fact_degrades_to_default(self) -> None:
        """Неизвестный факт → fact.default (0.0), исход неудовлетворён."""
        outcomes = (PreferredOutcome(_SERVICED, target=1.0),)
        pending = unsatisfied_outcomes(outcomes, {})
        assert len(pending) == 1

    def test_sorted_by_priority(self) -> None:
        """Порядок — по priority ↓, тай-брейк — порядок объявления."""
        low = PreferredOutcome(_SERVICED, priority=0.5)
        high = PreferredOutcome(_OTHER, priority=2.0)
        pending = unsatisfied_outcomes((low, high), {})
        assert pending == (high, low)

    def test_tie_break_declaration_order(self) -> None:
        """Равные приоритеты → порядок объявления."""
        first = PreferredOutcome(_SERVICED, priority=1.0)
        second = PreferredOutcome(_OTHER, priority=1.0)
        pending = unsatisfied_outcomes((first, second), {})
        assert pending == (first, second)


class TestSelectGoal:
    """Формирование цели из исходов и снимка фактов."""

    def test_all_satisfied_returns_none(self) -> None:
        """Все исходы удовлетворены → None."""
        outcomes = (PreferredOutcome(_SERVICED, target=1.0),)
        assert select_goal(outcomes, {"serviced": 1.0}) is None

    def test_empty_returns_none(self) -> None:
        """Пустой список исходов → None."""
        assert select_goal((), {}) is None

    def test_picks_highest_priority(self) -> None:
        """Первый неудовлетворённый по приоритету → Goal."""
        low = PreferredOutcome(_SERVICED, priority=0.5)
        high = PreferredOutcome(_OTHER, priority=2.0)
        goal = select_goal((low, high), {})
        assert goal == Goal(_OTHER, 1.0)

    def test_goal_uses_target(self) -> None:
        """Цель несёт target исхода."""
        outcome = PreferredOutcome(_SERVICED, target=0.8)
        goal = select_goal((outcome,), {"serviced": 0.3})
        assert goal == Goal(_SERVICED, 0.8)


class TestGoalReached:
    """Проверка достижения цели."""

    def test_reached(self) -> None:
        """fact >= value → True."""
        assert goal_reached(Goal(_SERVICED, 1.0), {"serviced": 1.0}) is True

    def test_not_reached(self) -> None:
        """fact < value → False."""
        assert goal_reached(Goal(_SERVICED, 1.0), {"serviced": 0.0}) is False

    def test_unknown_fact_uses_default(self) -> None:
        """Неизвестный факт → default (0.0) → не достигнута."""
        assert goal_reached(Goal(_SERVICED, 1.0), {}) is False


class TestEffectGain:
    """Доля закрытия разрыва (ADR-0015)."""

    def test_full_close(self) -> None:
        """effect.value=1 при current=0 → P=1."""
        option = _option("tool:a", effect=_SERVICED, effect_value=1.0)
        assert effect_gain(option, Goal(_SERVICED, 1.0), {"serviced": 0.0}) == 1.0

    def test_half_close(self) -> None:
        """effect.value=0.5 при current=0, goal=1 → P=0.5."""
        option = _option("tool:a", effect=_SERVICED, effect_value=0.5)
        assert effect_gain(option, Goal(_SERVICED, 1.0), {"serviced": 0.0}) == 0.5

    def test_effect_value_is_final_not_delta(self) -> None:
        """effect.value — итоговое значение: current=0.4, effect=0.4 → P=0."""
        option = _option("tool:a", effect=_SERVICED, effect_value=0.4)
        assert effect_gain(option, Goal(_SERVICED, 1.0), {"serviced": 0.4}) == 0.0

    def test_foreign_effect_zero(self) -> None:
        """Чужой факт эффекта → 0."""
        option = _option("tool:a", effect=_OTHER, effect_value=1.0)
        assert effect_gain(option, Goal(_SERVICED, 1.0), {}) == 0.0

    def test_no_effect_zero(self) -> None:
        """Нет эффекта → 0."""
        option = _option("tool:a")
        assert effect_gain(option, Goal(_SERVICED, 1.0), {}) == 0.0

    def test_goal_already_reached_zero(self) -> None:
        """Цель уже достигнута (gap_before <= ε) → 0."""
        option = _option("tool:a", effect=_SERVICED, effect_value=1.0)
        assert effect_gain(option, Goal(_SERVICED, 1.0), {"serviced": 1.0}) == 0.0

    def test_clipped_to_unit(self) -> None:
        """P не выходит за [0, 1]."""
        option = _option("tool:a", effect=_SERVICED, effect_value=1.0)
        value = effect_gain(option, Goal(_SERVICED, 1.0), {"serviced": -0.0})
        assert 0.0 <= value <= 1.0


class TestScoreOptionPragmatic:
    """Pragmatic в score_option — аддитивно, от цели."""

    def test_no_goal_zero(self) -> None:
        """Без context.goal pragmatic = 0 (совместимость этапов 1–5)."""
        option = _option("tool:a", effect=_SERVICED, effect_value=1.0)
        context = OptionContext(uncertainty=0.5)
        pragmatic, _, _ = score_option(option, context, ActuationPreferences())
        assert pragmatic == 0.0

    def test_with_goal_uses_effect_gain(self) -> None:
        """С context.goal pragmatic = effect_gain."""
        option = _option("tool:a", effect=_SERVICED, effect_value=1.0)
        context = OptionContext(
            uncertainty=0.5, goal=Goal(_SERVICED, 1.0), facts={"serviced": 0.0}
        )
        pragmatic, _, _ = score_option(option, context, ActuationPreferences())
        assert pragmatic == 1.0


class TestBackwardChainRanking:
    """Единый скаляр ранжирования и подцельный контекст."""

    def test_prefers_higher_pragmatic(self) -> None:
        """При context/preferences выбирается опция с большим pragmatic."""
        weak = _option("tool:weak", effect=_SERVICED, effect_value=0.3)
        strong = _option("tool:strong", effect=_SERVICED, effect_value=1.0)
        goal = Goal(_SERVICED, 1.0)
        context = OptionContext(uncertainty=0.0, goal=goal, facts={"serviced": 0.0})
        node = backward_chain(
            goal,
            (weak, strong),
            {"serviced": 0.0},
            context=context,
            preferences=ActuationPreferences(),
        )
        assert node.actuation is not None
        assert node.actuation.goal == "tool:strong"

    def test_without_context_uses_window_order(self) -> None:
        """Без context/preferences — первая по окну (прежнее поведение)."""
        weak = _option("tool:weak", effect=_SERVICED, effect_value=0.3)
        strong = _option("tool:strong", effect=_SERVICED, effect_value=1.0)
        node = backward_chain(Goal(_SERVICED, 1.0), (weak, strong), {"serviced": 0.0})
        assert node.actuation is not None
        assert node.actuation.goal == "tool:weak"

    def test_deterministic(self) -> None:
        """Одинаковый вход → одинаковое дерево."""
        weak = _option("tool:weak", effect=_SERVICED, effect_value=0.3)
        strong = _option("tool:strong", effect=_SERVICED, effect_value=1.0)
        goal = Goal(_SERVICED, 1.0)
        context = OptionContext(uncertainty=0.0, goal=goal, facts={})
        first = backward_chain(
            goal,
            (weak, strong),
            {},
            context=context,
            preferences=ActuationPreferences(),
        )
        second = backward_chain(
            goal,
            (weak, strong),
            {},
            context=context,
            preferences=ActuationPreferences(),
        )
        assert first == second
