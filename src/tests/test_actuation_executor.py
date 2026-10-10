"""Tests for actuation effectors and executor (S8 stage 6, ADR-0012 §7–8).

Covers the deferred effector contract (no blocking, preempt), tool/speech
adapters, and ``ActuatorExecutor``: a step completes on the next tick, preemption
aborts the running step, results reach ``completed`` (bus), impatience grows with
waiting, ``done`` is terminal and the tree is not replayed, and runs are
deterministic.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from src.core.actuation import (
    Actuation,
    ActuationKind,
    ActuationResult,
    ActuationStatus,
    Fact,
    Guard,
    Node,
    NodeKind,
    NodeStatus,
)
from src.host.effectors import (
    DeferredEffector,
    Effector,
    SpeechEffector,
    ToolEffector,
)
from src.host.executor import ActuatorExecutor, node_at
from src.host.gate import Capability, CapabilityGate, CapabilityTier
from src.host.probe import ProbeEffector
from src.mcp import Affordance, AffordanceMap, SignalCategory


def _ok(_a: Actuation) -> ActuationResult:
    """Работа, всегда успешная."""
    return ActuationResult(ActuationStatus.SUCCESS, (1.0,))


def _action(goal: str, kind: ActuationKind = ActuationKind.INVOKE_TOOL) -> Node:
    """Собрать Action-лист."""
    return Node(NodeKind.ACTION, name=goal, actuation=Actuation(kind, goal, goal))


class TestDeferredEffector:
    """Tests for the deferred effector contract."""

    def test_no_work_before_latency(self) -> None:
        """Работа не выполняется до истечения latency (нет блокировки)."""
        calls: list[str] = []

        def work(a: Actuation) -> ActuationResult:
            calls.append(a.goal)
            return ActuationResult(ActuationStatus.SUCCESS)

        effector = DeferredEffector(work, latency_ticks=2)
        effector.start(Actuation(ActuationKind.SPEAK, "g", "p"))
        assert effector.poll().status is ActuationStatus.RUNNING
        assert calls == []
        assert effector.poll().status is ActuationStatus.RUNNING
        assert calls == []
        assert effector.poll().status is ActuationStatus.SUCCESS
        assert calls == ["g"]

    def test_result_cached(self) -> None:
        """Результат вычисляется один раз и кэшируется."""
        calls: list[int] = []

        def work(_a: Actuation) -> ActuationResult:
            calls.append(1)
            return ActuationResult(ActuationStatus.SUCCESS)

        effector = DeferredEffector(work)
        effector.start(Actuation(ActuationKind.SPEAK, "g", "p"))
        effector.poll()
        effector.poll()
        assert len(calls) == 1

    def test_goal_reflects_state(self) -> None:
        """goal — исполняемая активация или None."""
        effector = DeferredEffector(_ok)
        assert effector.goal is None
        effector.start(Actuation(ActuationKind.SPEAK, "g", "p"))
        assert effector.goal == "g"

    def test_preempt_resets(self) -> None:
        """preempt → Preempted, активация сброшена."""
        effector = DeferredEffector(_ok, latency_ticks=1)
        effector.start(Actuation(ActuationKind.SPEAK, "g", "p"))
        effector.preempt()
        assert effector.goal is None
        assert effector.poll().status is ActuationStatus.PREEMPTED

    def test_work_error_is_failure(self) -> None:
        """Сбой работы → Failure (не роняет)."""

        def boom(_a: Actuation) -> ActuationResult:
            raise RuntimeError("boom")

        effector = DeferredEffector(boom)
        effector.start(Actuation(ActuationKind.SPEAK, "g", "p"))
        assert effector.poll().status is ActuationStatus.FAILURE

    def test_negative_latency_rejected(self) -> None:
        """latency_ticks < 0 → ValueError."""
        with pytest.raises(ValueError, match="latency_ticks must be >= 0"):
            DeferredEffector(_ok, latency_ticks=-1)

    def test_satisfies_protocol(self) -> None:
        """DeferredEffector удовлетворяет Effector Protocol."""
        assert isinstance(DeferredEffector(_ok), Effector)


class TestAdapters:
    """Tests for ToolEffector and SpeechEffector."""

    def _probe(self, *, granted: frozenset[Capability] | None = None) -> ProbeEffector:
        amap = AffordanceMap(
            (Affordance("web_search", SignalCategory.EXTEROCEPTIVE, True, 3),)
        )
        return ProbeEffector(
            affordances=amap,
            gate=CapabilityGate(
                max_tier=CapabilityTier.T4,
                granted=granted
                if granted is not None
                else frozenset({Capability.READ, Capability.ACT_REVERSIBLE}),
            ),
        )

    def test_tool_success(self) -> None:
        """Успешный тул → Success с данными."""
        effector = ToolEffector(self._probe())
        effector.start(
            Actuation(ActuationKind.INVOKE_TOOL, "tool:web_search", "web_search")
        )
        result = effector.poll()
        assert result.status is ActuationStatus.SUCCESS
        assert len(result.data) == 3

    def test_tool_denied_is_failure(self) -> None:
        """Отказ гейта → Failure."""
        effector = ToolEffector(self._probe(granted=frozenset({Capability.READ})))
        effector.start(
            Actuation(ActuationKind.INVOKE_TOOL, "tool:web_search", "web_search")
        )
        assert effector.poll().status is ActuationStatus.FAILURE

    def test_speech_success_and_failure(self) -> None:
        """Речь: текст → Success, None → Failure."""
        effector = SpeechEffector(lambda _a: "hello")
        effector.start(Actuation(ActuationKind.SPEAK, "say", "hi"))
        assert effector.poll().status is ActuationStatus.SUCCESS

        silent = SpeechEffector(lambda _a: None)
        silent.start(Actuation(ActuationKind.SPEAK, "say", "hi"))
        assert silent.poll().status is ActuationStatus.FAILURE


class TestNodeAt:
    """Tests for node_at path resolution."""

    def test_root_path(self) -> None:
        """Пустой путь — сам корень."""
        root = _action("a")
        assert node_at(root, ()) is root

    def test_nested_path(self) -> None:
        """Путь до вложенного узла."""
        inner = _action("b")
        root = Node(NodeKind.SEQUENCE, children=(_action("a"), inner))
        assert node_at(root, (1,)) is inner

    def test_bad_path_raises(self) -> None:
        """Неверный путь → IndexError."""
        with pytest.raises(IndexError):
            node_at(_action("a"), (0,))


def _executor(
    *,
    latency_ticks: int = 0,
    expected_ticks: int = 1,
    work: Callable[[Actuation], ActuationResult] = _ok,
) -> ActuatorExecutor:
    """Executor с одним тул-эффектором для теста."""
    effector = DeferredEffector(work, latency_ticks=latency_ticks)
    return ActuatorExecutor(
        effectors={ActuationKind.INVOKE_TOOL: effector},
        expected_ticks=expected_ticks,
    )


class TestActuatorExecutor:
    """Tests for the BT-driving executor."""

    def test_step_completes_next_tick(self) -> None:
        """Шаг не блокирует тик: завершение приходит на следующем тике."""
        ex = _executor(latency_ticks=0)
        root = _action("a")
        first = ex.tick(root, {})
        assert first.status is NodeStatus.RUNNING
        assert first.running_goal == "a"
        second = ex.tick(root, {})
        assert second.status is NodeStatus.SUCCESS
        assert ex.done

    def test_latency_defers_completion(self) -> None:
        """latency_ticks откладывает завершение."""
        ex = _executor(latency_ticks=2)
        root = _action("a")
        assert ex.tick(root, {}).status is NodeStatus.RUNNING
        assert ex.tick(root, {}).status is NodeStatus.RUNNING
        assert ex.tick(root, {}).status is NodeStatus.RUNNING
        assert ex.tick(root, {}).status is NodeStatus.SUCCESS

    def test_completed_reaches_outcome(self) -> None:
        """Завершённая активация попадает в completed (→ шина)."""
        ex = _executor()
        root = _action("a")
        ex.tick(root, {})
        outcome = ex.tick(root, {})
        assert len(outcome.completed) == 1
        assert outcome.completed[0].status is ActuationStatus.SUCCESS

    def test_done_is_terminal(self) -> None:
        """После Success дерево не переигрывается."""
        ex = _executor()
        root = _action("a")
        ex.tick(root, {})
        final = ex.tick(root, {})
        assert final.status is NodeStatus.SUCCESS
        replay = ex.tick(root, {})
        assert replay.status is NodeStatus.SUCCESS
        assert replay.completed == ()

    def test_sequence_two_steps(self) -> None:
        """Последовательность из двух шагов проходит по порядку."""
        ex = _executor()
        root = Node(NodeKind.SEQUENCE, children=(_action("a"), _action("b")))
        assert ex.tick(root, {}).running_goal == "a"
        assert ex.tick(root, {}).running_goal == "b"
        assert ex.tick(root, {}).status is NodeStatus.SUCCESS

    def test_preemption_on_goal_change(self) -> None:
        """Смена дерева прерывает бегущий шаг (fail-safe, Preempted)."""
        ex = _executor(latency_ticks=5)
        first_root = _action("a")
        ex.tick(first_root, {})
        # Новое дерево с другой целью → прежний эффектор прерван.
        second_root = _action("b")
        outcome = ex.tick(second_root, {})
        assert outcome.running_goal == "b"

    def test_guard_fallback_skips_blocked(self) -> None:
        """Fallback: заблокированный guard пропускается, идёт действие."""
        net = Fact("network_available")
        root = Node(
            NodeKind.FALLBACK,
            children=(
                Node(NodeKind.CONDITION, name="net", guard=Guard(net, 0.5)),
                _action("b"),
            ),
        )
        ex = _executor()
        blocked = ex.tick(root, {"network_available": 0.0})
        assert blocked.running_goal == "b"
        done = ex.tick(root, {"network_available": 0.0})
        assert done.status is NodeStatus.SUCCESS

    def test_impatience_grows_with_waiting(self) -> None:
        """Нетерпение растёт со временем ожидания (сигнал)."""
        ex = _executor(latency_ticks=10, expected_ticks=1)
        root = _action("a")
        assert ex.tick(root, {}).impatience == 0.0
        assert ex.tick(root, {}).impatience == 0.0  # elapsed=1, expected=1
        assert ex.tick(root, {}).impatience > 0.0
        assert ex.tick(root, {}).impatience <= 1.0

    def test_no_effector_for_kind_fails(self) -> None:
        """Нет эффектора под вид → честный Failure."""
        ex = ActuatorExecutor(effectors={})
        outcome = ex.tick(_action("a"), {})
        assert outcome.status is NodeStatus.FAILURE
        assert ex.done

    def test_determinism(self) -> None:
        """Одинаковый вход → одинаковая последовательность исходов."""
        root = Node(NodeKind.SEQUENCE, children=(_action("a"), _action("b")))

        def run() -> list[tuple[NodeStatus, str | None]]:
            ex = _executor()
            return [
                (o.status, o.running_goal)
                for o in (ex.tick(root, {}), ex.tick(root, {}), ex.tick(root, {}))
            ]

        assert run() == run()

    def test_expected_ticks_validation(self) -> None:
        """expected_ticks < 0 → ValueError."""
        with pytest.raises(ValueError, match="expected_ticks must be >= 0"):
            ActuatorExecutor(effectors={}, expected_ticks=-1)
