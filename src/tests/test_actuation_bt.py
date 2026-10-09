"""Unit tests for the actuation Behavior Tree (S8 stage 4, ADR-0012).

Covers node validation, leaf evaluation (``CONDITION``/``ACTION``), ``SEQUENCE``
ordering and failure, ``FALLBACK`` priority and preemption, ``order_children``
soft ordering, reactivity (re-evaluation from the root) and determinism.
"""

from __future__ import annotations

import pytest

from src.core.actuation import (
    Actuation,
    ActuationKind,
    Fact,
    Guard,
    Node,
    NodeKind,
    NodeStatus,
    Regularity,
    TickContext,
    TickMemory,
    order_children,
    tick,
)


def _guard(name: str = "network_available", threshold: float = 0.5) -> Guard:
    """Собрать guard для теста."""
    return Guard(Fact(name=name), threshold=threshold)


def _action(goal: str) -> Node:
    """Собрать лист-действие для теста."""
    return Node(
        NodeKind.ACTION,
        name=goal,
        actuation=Actuation(ActuationKind.INVOKE_TOOL, goal, goal),
    )


def _condition(name: str = "network_available", threshold: float = 0.5) -> Node:
    """Собрать лист-условие для теста."""
    return Node(NodeKind.CONDITION, name=name, guard=_guard(name, threshold))


class TestNodeValidation:
    """Tests for Node shape validation."""

    def test_condition_requires_guard(self) -> None:
        """CONDITION без guard → ValueError."""
        with pytest.raises(ValueError, match="condition node requires a guard"):
            Node(NodeKind.CONDITION)

    def test_action_requires_actuation(self) -> None:
        """ACTION без actuation → ValueError."""
        with pytest.raises(ValueError, match="action node requires an actuation"):
            Node(NodeKind.ACTION)

    def test_leaf_must_not_have_children(self) -> None:
        """Лист с детьми → ValueError."""
        with pytest.raises(ValueError, match="must not have children"):
            Node(NodeKind.CONDITION, guard=_guard(), children=(_action("x"),))

    def test_composite_requires_children(self) -> None:
        """Композит без детей → ValueError."""
        with pytest.raises(ValueError, match="requires children"):
            Node(NodeKind.SEQUENCE)

    def test_composite_must_not_carry_guard(self) -> None:
        """Композит с guard → ValueError."""
        with pytest.raises(ValueError, match="must not carry guard/actuation"):
            Node(NodeKind.SEQUENCE, guard=_guard(), children=(_action("x"),))

    def test_valid_composite(self) -> None:
        """Композит с детьми валиден."""
        node = Node(NodeKind.SEQUENCE, children=(_action("x"),))
        assert node.children[0].kind is NodeKind.ACTION


class TestLeaves:
    """Tests for CONDITION and ACTION leaves."""

    def test_condition_true(self) -> None:
        """Факт выше порога → SUCCESS."""
        node = _condition(threshold=0.5)
        status, memory = tick(node, TickContext(facts={"network_available": 0.9}))
        assert status is NodeStatus.SUCCESS
        assert memory == TickMemory()

    def test_condition_false(self) -> None:
        """Факт ниже порога → FAILURE."""
        node = _condition(threshold=0.5)
        status, _ = tick(node, TickContext(facts={"network_available": 0.1}))
        assert status is NodeStatus.FAILURE

    def test_condition_unknown_fact_uses_default(self) -> None:
        """Неизвестный факт → дефолт (0.0), порог не взят → FAILURE."""
        node = _condition(threshold=0.5)
        status, _ = tick(node, TickContext())
        assert status is NodeStatus.FAILURE

    def test_action_default_running(self) -> None:
        """Действие без отчёта Shell → RUNNING."""
        node = _action("tool:x")
        status, _ = tick(node, TickContext())
        assert status is NodeStatus.RUNNING

    def test_action_reports_success(self) -> None:
        """Отчёт Shell SUCCESS → SUCCESS."""
        node = _action("tool:x")
        status, _ = tick(
            node, TickContext(action_status={"tool:x": NodeStatus.SUCCESS})
        )
        assert status is NodeStatus.SUCCESS

    def test_action_reports_failure(self) -> None:
        """Отчёт Shell FAILURE → FAILURE."""
        node = _action("tool:x")
        status, _ = tick(
            node, TickContext(action_status={"tool:x": NodeStatus.FAILURE})
        )
        assert status is NodeStatus.FAILURE


class TestSequence:
    """Tests for SEQUENCE."""

    def test_all_success(self) -> None:
        """Все дети успешны → SUCCESS."""
        node = Node(NodeKind.SEQUENCE, children=(_action("a"), _action("b")))
        context = TickContext(
            action_status={"a": NodeStatus.SUCCESS, "b": NodeStatus.SUCCESS}
        )
        status, memory = tick(node, context)
        assert status is NodeStatus.SUCCESS
        assert memory == TickMemory()

    def test_first_failure_stops(self) -> None:
        """Первый провал останавливает → FAILURE."""
        node = Node(NodeKind.SEQUENCE, children=(_action("a"), _action("b")))
        context = TickContext(
            action_status={"a": NodeStatus.FAILURE, "b": NodeStatus.SUCCESS}
        )
        status, _ = tick(node, context)
        assert status is NodeStatus.FAILURE

    def test_running_child_reported(self) -> None:
        """Бегущий ребёнок → RUNNING с путём до него."""
        node = Node(NodeKind.SEQUENCE, children=(_action("a"), _action("b")))
        context = TickContext(
            action_status={"a": NodeStatus.SUCCESS, "b": NodeStatus.RUNNING}
        )
        status, memory = tick(node, context)
        assert status is NodeStatus.RUNNING
        assert memory.running_path == (1,)

    def test_nested_running_path(self) -> None:
        """Путь до вложенного бегущего листа накапливается."""
        inner = Node(NodeKind.SEQUENCE, children=(_action("b"),))
        outer = Node(NodeKind.SEQUENCE, children=(_action("a"), inner))
        context = TickContext(
            action_status={"a": NodeStatus.SUCCESS, "b": NodeStatus.RUNNING}
        )
        status, memory = tick(outer, context)
        assert status is NodeStatus.RUNNING
        assert memory.running_path == (1, 0)

    def test_resume_via_action_status(self) -> None:
        """Завершённое действие (SUCCESS) продолжает последовательность."""
        node = Node(NodeKind.SEQUENCE, children=(_action("a"), _action("b")))
        # Тик 1: a бежит.
        status1, _ = tick(node, TickContext())
        assert status1 is NodeStatus.RUNNING
        # Тик 2: Shell отчитался о a → идём к b.
        status2, memory2 = tick(
            node, TickContext(action_status={"a": NodeStatus.SUCCESS})
        )
        assert status2 is NodeStatus.RUNNING
        assert memory2.running_path == (1,)


class TestFallback:
    """Tests for FALLBACK priority and preemption."""

    def test_first_success_wins(self) -> None:
        """Первый успешный ребёнок → SUCCESS."""
        node = Node(NodeKind.FALLBACK, children=(_action("a"), _action("b")))
        context = TickContext(
            action_status={"a": NodeStatus.SUCCESS, "b": NodeStatus.SUCCESS}
        )
        status, _ = tick(node, context)
        assert status is NodeStatus.SUCCESS

    def test_all_failure(self) -> None:
        """Все дети провалились → FAILURE."""
        node = Node(NodeKind.FALLBACK, children=(_action("a"), _action("b")))
        context = TickContext(
            action_status={"a": NodeStatus.FAILURE, "b": NodeStatus.FAILURE}
        )
        status, _ = tick(node, context)
        assert status is NodeStatus.FAILURE

    def test_falls_through_to_running(self) -> None:
        """Первый провал → пробуем следующего (RUNNING)."""
        node = Node(NodeKind.FALLBACK, children=(_action("a"), _action("b")))
        context = TickContext(action_status={"a": NodeStatus.FAILURE})
        status, memory = tick(node, context)
        assert status is NodeStatus.RUNNING
        assert memory.running_path == (1,)

    def test_preemption_by_priority(self) -> None:
        """Более приоритетный ребёнок вытесняет бегущего (преемпция)."""
        # Сначала a провалился → бежит b (путь 1); затем a стал RUNNING →
        # вытесняет b (путь 0).
        node = Node(NodeKind.FALLBACK, children=(_action("a"), _action("b")))
        before = TickContext(
            action_status={"a": NodeStatus.FAILURE, "b": NodeStatus.RUNNING}
        )
        status_before, memory_before = tick(node, before)
        assert status_before is NodeStatus.RUNNING
        assert memory_before.running_path == (1,)
        after = TickContext(
            action_status={"a": NodeStatus.RUNNING, "b": NodeStatus.RUNNING}
        )
        status_after, memory_after = tick(node, after)
        assert status_after is NodeStatus.RUNNING
        assert memory_after.running_path == (0,)


class TestOrderChildren:
    """Tests for order_children — soft ordering by regularity cost."""

    def test_orders_by_cost(self) -> None:
        """Меньшая стоимость — раньше (приоритетнее)."""
        cheap = Node(
            NodeKind.ACTION,
            name="cheap",
            actuation=Actuation(ActuationKind.INVOKE_TOOL, "cheap", "cheap"),
            regularities=(Regularity(Fact("network_available"), weight=0.0),),
        )
        pricey = Node(
            NodeKind.ACTION,
            name="pricey",
            actuation=Actuation(ActuationKind.INVOKE_TOOL, "pricey", "pricey"),
            regularities=(Regularity(Fact("network_available"), weight=1.0),),
        )
        ordered = order_children((pricey, cheap), {"network_available": 1.0})
        assert [n.name for n in ordered] == ["cheap", "pricey"]

    def test_stable_tie_break(self) -> None:
        """Равная стоимость → исходный порядок (детерминизм)."""
        a = _action("a")
        b = _action("b")
        ordered = order_children((a, b), {})
        assert [n.name for n in ordered] == ["a", "b"]

    def test_fallback_uses_soft_order(self) -> None:
        """FALLBACK учитывает мягкий порядок детей."""
        pricey = Node(
            NodeKind.ACTION,
            name="pricey",
            actuation=Actuation(ActuationKind.INVOKE_TOOL, "pricey", "pricey"),
            regularities=(Regularity(Fact("network_available"), weight=1.0),),
        )
        cheap = Node(
            NodeKind.ACTION,
            name="cheap",
            actuation=Actuation(ActuationKind.INVOKE_TOOL, "cheap", "cheap"),
            regularities=(Regularity(Fact("network_available"), weight=0.0),),
        )
        node = Node(NodeKind.FALLBACK, children=(pricey, cheap))
        status, memory = tick(node, TickContext(facts={"network_available": 1.0}))
        assert status is NodeStatus.RUNNING
        assert memory.running_path == (0,)  # cheap стал первым


class TestReactivityAndDeterminism:
    """Tests for reactivity and determinism of tick."""

    def test_reactivity_re_evaluates(self) -> None:
        """Дерево перерешается по текущему контексту (реактивность)."""
        node = Node(NodeKind.FALLBACK, children=(_condition(), _action("b")))
        blocked = tick(node, TickContext(facts={"network_available": 0.0}))
        assert blocked[0] is NodeStatus.RUNNING
        allowed = tick(node, TickContext(facts={"network_available": 1.0}))
        assert allowed[0] is NodeStatus.SUCCESS

    def test_determinism(self) -> None:
        """Одинаковый вход → одинаковый выход."""
        node = Node(NodeKind.FALLBACK, children=(_condition(), _action("b")))
        context = TickContext(facts={"network_available": 0.0})
        assert tick(node, context) == tick(node, context)
