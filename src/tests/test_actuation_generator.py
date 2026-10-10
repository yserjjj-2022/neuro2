"""Unit tests for the actuation runtime generator (S8 stage 5, ADR-0012).

Covers ``Goal`` validation, backward chaining (single-step and two-step via
guard), the "already true" case, safe refusal (unreachable / exhausted horizon),
``max_depth`` truncation, determinism and the ablation "remove effect → the
chain collapses".
"""

from __future__ import annotations

import pytest

from src.core.actuation import (
    Effect,
    Fact,
    Goal,
    Guard,
    NodeKind,
    NodeStatus,
    Option,
    OptionSource,
    TickContext,
    backward_chain,
    tick,
)

_NET = Fact("network_available")
_TOPIC = Fact("topic_bound")


def _option(
    option_id: str,
    *,
    effect: Fact | None,
    guard: Guard | None = None,
    source: OptionSource = OptionSource.TOOL,
) -> Option:
    """Собрать опцию для теста генератора."""
    return Option(
        option_id,
        source,
        guard=guard,
        effect=None if effect is None else Effect(effect),
    )


class TestGoal:
    """Tests for Goal validation."""

    def test_default_value(self) -> None:
        """Дефолтное целевое значение — 1."""
        assert Goal(_NET).value == 1.0

    def test_value_above_one_rejected(self) -> None:
        """value > 1 → ValueError."""
        with pytest.raises(ValueError, match="goal value must be in"):
            Goal(_NET, value=1.5)

    def test_value_below_zero_rejected(self) -> None:
        """value < 0 → ValueError."""
        with pytest.raises(ValueError, match="goal value must be in"):
            Goal(_NET, value=-0.1)


class TestBackwardChain:
    """Tests for backward_chain tree derivation."""

    def test_max_depth_validation(self) -> None:
        """max_depth < 1 → ValueError."""
        with pytest.raises(ValueError, match="max_depth must be >= 1"):
            backward_chain(Goal(_NET), (), {}, max_depth=0)

    def test_already_true_is_condition(self) -> None:
        """Цель уже истинна → Condition-узел (без действия)."""
        node = backward_chain(Goal(_NET), (), {"network_available": 1.0})
        assert node.kind is NodeKind.CONDITION
        status, _ = tick(node, TickContext(facts={"network_available": 1.0}))
        assert status is NodeStatus.SUCCESS

    def test_single_step_is_action(self) -> None:
        """Опция без guard → Action-узел."""
        options = (_option("tool:probe", effect=_NET),)
        node = backward_chain(Goal(_NET), options, {})
        assert node.kind is NodeKind.ACTION
        assert node.actuation is not None
        assert node.actuation.goal == "tool:probe"

    def test_payload_falls_back_to_id_without_target(self) -> None:
        """Без target payload = id (обратная совместимость hand-built опций)."""
        options = (_option("tool:probe", effect=_NET),)
        node = backward_chain(Goal(_NET), options, {})
        assert node.actuation is not None
        assert node.actuation.payload == "tool:probe"

    def test_payload_uses_target_when_set(self) -> None:
        """С target payload = реальное имя вызова, а goal — id окна."""
        options = (
            Option("tool:probe", OptionSource.TOOL, target="probe", effect=Effect(_NET)),
        )
        node = backward_chain(Goal(_NET), options, {})
        assert node.actuation is not None
        assert node.actuation.goal == "tool:probe"
        assert node.actuation.payload == "probe"

    def test_two_step_via_guard(self) -> None:
        """Опция с guard → Sequence(подцель, действие)."""
        options = (
            _option("tool:probe", effect=_NET, guard=Guard(_TOPIC, 0.5)),
            _option("tool:search", effect=_TOPIC),
        )
        node = backward_chain(Goal(_NET), options, {})
        assert node.kind is NodeKind.SEQUENCE
        assert len(node.children) == 2
        # Первый ребёнок — подцель (Action по topic_bound), второй — действие.
        assert node.children[0].kind is NodeKind.ACTION
        assert node.children[0].actuation is not None
        assert node.children[0].actuation.goal == "tool:search"
        assert node.children[1].actuation is not None
        assert node.children[1].actuation.goal == "tool:probe"

    def test_two_step_chain_executes_in_order(self) -> None:
        """Выведенная цепочка проходит шаги по порядку через action_status."""
        options = (
            _option("tool:probe", effect=_NET, guard=Guard(_TOPIC, 0.5)),
            _option("tool:search", effect=_TOPIC),
        )
        node = backward_chain(Goal(_NET), options, {})
        # Шаг 1: search бежит.
        status1, memory1 = tick(node, TickContext())
        assert status1 is NodeStatus.RUNNING
        assert memory1.running_path == (0,)
        # Шаг 2: search завершён → probe бежит.
        status2, memory2 = tick(
            node, TickContext(action_status={"tool:search": NodeStatus.SUCCESS})
        )
        assert status2 is NodeStatus.RUNNING
        assert memory2.running_path == (1,)

    def test_unreachable_goal_is_safe_refusal(self) -> None:
        """Нет опции, достигающей факта → Condition (честный отказ)."""
        options = (_option("tool:other", effect=_TOPIC),)
        node = backward_chain(Goal(_NET), options, {})
        assert node.kind is NodeKind.CONDITION
        status, _ = tick(node, TickContext())
        assert status is NodeStatus.FAILURE

    def test_no_fabricated_action(self) -> None:
        """Отказ не выдумывает действие (нет Action-узла)."""
        node = backward_chain(Goal(_NET), (), {})
        assert node.kind is NodeKind.CONDITION
        assert node.actuation is None

    def test_self_loop_terminates_at_horizon(self) -> None:
        """Guard, ведущий к тому же факту, упирается в горизонт."""
        options = (_option("tool:x", effect=_NET, guard=Guard(_NET, 0.5)),)
        node = backward_chain(Goal(_NET), options, {}, max_depth=2)
        # Дерево конечной глубины, не бесконечная рекурсия.
        assert node.kind is NodeKind.SEQUENCE

    def test_max_depth_truncates(self) -> None:
        """max_depth обрезает цепочку (горизонт)."""
        options = (
            _option("tool:a", effect=_NET, guard=Guard(_TOPIC, 0.5)),
            _option("tool:b", effect=_TOPIC),
        )
        shallow = backward_chain(Goal(_NET), options, {}, max_depth=1)
        deep = backward_chain(Goal(_NET), options, {}, max_depth=3)
        # Глубокое дерево содержит подцель как Action; мелкое — обрезано.
        assert shallow.children[0].kind is NodeKind.CONDITION
        assert deep.children[0].kind is NodeKind.ACTION

    def test_tie_break_window_order(self) -> None:
        """При нескольких подходящих опциях берётся первая в окне."""
        options = (
            _option("tool:first", effect=_NET),
            _option("tool:second", effect=_NET),
        )
        node = backward_chain(Goal(_NET), options, {})
        assert node.actuation is not None
        assert node.actuation.goal == "tool:first"

    def test_determinism(self) -> None:
        """Одинаковый вход → одинаковое дерево."""
        options = (
            _option("tool:probe", effect=_NET, guard=Guard(_TOPIC, 0.5)),
            _option("tool:search", effect=_TOPIC),
        )
        goal = Goal(_NET)
        assert backward_chain(goal, options, {}) == backward_chain(goal, options, {})

    def test_ablation_without_effect_collapses(self) -> None:
        """Убрать effect → цепочка схлопывается до Condition (ablation)."""
        with_effect = (
            _option("tool:probe", effect=_NET, guard=Guard(_TOPIC, 0.5)),
            _option("tool:search", effect=_TOPIC),
        )
        without_effect = (
            _option("tool:probe", effect=None, guard=Guard(_TOPIC, 0.5)),
            _option("tool:search", effect=_TOPIC),
        )
        assert backward_chain(Goal(_NET), with_effect, {}).kind is NodeKind.SEQUENCE
        collapsed = backward_chain(Goal(_NET), without_effect, {})
        assert collapsed.kind is NodeKind.CONDITION
