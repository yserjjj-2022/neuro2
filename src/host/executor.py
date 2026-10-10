"""ActuatorExecutor — Imperative Shell driving a BT over time (S8 stage 6).

Reactive by construction (ADR-0012 §5): the tree is re-ticked from the root each
tick; action outcomes come from effectors (never blocking the tick). Completed
actuations are surfaced for the bus (ADR-0012 §7); preemption aborts a running
step when the tree selects a different action (fail-safe: a preempted step is
never successful); waiting is a signal (``impatience``, ADR-0012 §8).

Determinism: with fixed effector ``latency_ticks`` a run is reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.core.actuation import (
    Actuation,
    ActuationKind,
    ActuationResult,
    ActuationStatus,
    Node,
    NodeKind,
    NodeStatus,
    TickContext,
    tick,
)
from src.host.effectors import Effector

# Терминальные статусы активации (эффектор отчитался).
_TERMINAL = frozenset(
    {ActuationStatus.SUCCESS, ActuationStatus.FAILURE, ActuationStatus.PREEMPTED}
)


@dataclass(frozen=True)
class ExecutorOutcome:
    """Итог одного тика executor'а (S8 этап 6).

    Attributes:
        status: Статус корня дерева.
        running_goal: Идентификатор бегущей активации (None — ничего не бежит).
        impatience: Сигнал нетерпения ∈ [0, 1] (elapsed против ожидания).
        completed: Завершившиеся в этом тике активации → в шину (ADR-0012 §7).
        preemptions: Сколько бегущих шагов прервано в этом тике (fail-safe).
    """

    status: NodeStatus
    running_goal: str | None
    impatience: float
    completed: tuple[ActuationResult, ...]
    preemptions: int = 0


def node_at(root: Node, path: tuple[int, ...]) -> Node:
    """Узел по пути от корня (чистая, этап 6).

    Args:
        root: Корень дерева.
        path: Индексы детей от корня (пустой путь — сам корень).

    Returns:
        Узел по пути.

    Raises:
        IndexError: Если путь не соответствует структуре дерева.
    """
    node = root
    for index in path:
        node = node.children[index]
    return node


def _to_node_status(status: ActuationStatus) -> NodeStatus:
    """Перевести статус активации в статус узла (``Preempted`` → ``Failure``)."""
    return (
        NodeStatus.SUCCESS if status is ActuationStatus.SUCCESS else NodeStatus.FAILURE
    )


class ActuatorExecutor:
    """Ведёт BT-дерево во времени через эффекторы (Shell, S8 этап 6).

    Attributes:
        effectors: Эффекторы по виду актуации.
        expected_ticks: Ожидаемая длительность шага (для нетерпения).
    """

    def __init__(
        self,
        *,
        effectors: Mapping[ActuationKind, Effector],
        expected_ticks: int = 1,
    ) -> None:
        """Создать executor.

        Args:
            effectors: Эффекторы по виду актуации (речь/тул).
            expected_ticks: Ожидаемая длительность шага в тиках (>= 0).

        Raises:
            ValueError: Если ``expected_ticks < 0``.
        """
        if expected_ticks < 0:
            raise ValueError(f"expected_ticks must be >= 0, got {expected_ticks}")
        self.effectors: dict[ActuationKind, Effector] = dict(effectors)
        self.expected_ticks = expected_ticks
        self._root: Node | None = None
        self._statuses: dict[str, NodeStatus] = {}
        self._done = False
        self._final = NodeStatus.FAILURE
        self._target_goal: str | None = None
        self._elapsed = 0
        self._preemptions = 0

    @property
    def done(self) -> bool:
        """Завершено ли дерево (``Success``/``Failure``)."""
        return self._done

    def _reset(self, root: Node) -> int:
        """Сбросить состояние при смене дерева (преемпция, fail-safe).

        Returns:
            Сколько бегущих шагов прервано сбросом.
        """
        preemptions = 0
        for effector in self.effectors.values():
            if effector.goal is not None:
                effector.preempt()
                preemptions += 1
        self._root = root
        self._statuses = {}
        self._done = False
        self._final = NodeStatus.FAILURE
        self._target_goal = None
        self._elapsed = 0
        return preemptions

    def _poll_effectors(self) -> list[ActuationResult]:
        """Опросить эффекторы; вернуть завершившиеся в этом тике (ADR-0012 §7)."""
        completed: list[ActuationResult] = []
        for effector in self.effectors.values():
            goal = effector.goal
            if goal is None:
                continue
            result = effector.poll()
            if result.status in _TERMINAL and goal not in self._statuses:
                self._statuses[goal] = _to_node_status(result.status)
                completed.append(result)
        return completed

    def _apply_preemption(self, target_goal: str | None) -> int:
        """Прервать эффекторы, исполняющие не целевую активацию (fail-safe).

        Returns:
            Сколько бегущих шагов прервано.
        """
        preemptions = 0
        for effector in self.effectors.values():
            if effector.goal is not None and effector.goal != target_goal:
                effector.preempt()
                preemptions += 1
        return preemptions

    def _impatience(self, target_goal: str | None) -> float:
        """Сигнал нетерпения: elapsed против ожидания (ADR-0012 §8)."""
        if target_goal is None:
            self._target_goal = None
            self._elapsed = 0
            return 0.0
        if target_goal == self._target_goal:
            self._elapsed += 1
        else:
            self._target_goal = target_goal
            self._elapsed = 0
        if self.expected_ticks <= 0:
            return 0.0
        deviation = (self._elapsed - self.expected_ticks) / self.expected_ticks
        return min(1.0, max(0.0, deviation))

    def tick(self, root: Node, facts: Mapping[str, float]) -> ExecutorOutcome:
        """Провести один тик: опрос → tick BT → старт/преемпция (S8 этап 6).

        Args:
            root: Корень дерева текущего deliberative-цикла.
            facts: Снимок фактов мира (для кондишенов).

        Returns:
            ExecutorOutcome — статус, бегущая цель, нетерпение и завершённые
            активации этого тика.
        """
        if self._root is None or self._root != root:
            self._reset(root)
        if self._done:
            return ExecutorOutcome(self._final, None, 0.0, ())

        completed = self._poll_effectors()
        context = TickContext(facts=facts, action_status=dict(self._statuses))
        status, memory = tick(root, context)

        if status is not NodeStatus.RUNNING:
            self._done = True
            self._final = status
            preemptions = self._apply_preemption(None)
            return ExecutorOutcome(status, None, 0.0, tuple(completed), preemptions)

        running = node_at(root, memory.running_path)
        if running.kind is not NodeKind.ACTION or running.actuation is None:
            # Защита: бегущий узел должен быть Action-листом (валидация Node).
            self._done = True
            self._final = NodeStatus.FAILURE
            preemptions = self._apply_preemption(None)
            return ExecutorOutcome(
                NodeStatus.FAILURE, None, 0.0, tuple(completed), preemptions
            )

        target: Actuation = running.actuation
        impatience = self._impatience(target.goal)
        effector = self.effectors.get(target.kind)
        if effector is None:
            # Нет эффектора под вид — честный провал, не выдумываем исполнение.
            self._statuses[target.goal] = NodeStatus.FAILURE
            self._done = True
            self._final = NodeStatus.FAILURE
            preemptions = self._apply_preemption(None)
            return ExecutorOutcome(
                NodeStatus.FAILURE, None, 0.0, tuple(completed), preemptions
            )

        preemptions = self._apply_preemption(target.goal)
        if effector.goal != target.goal:
            effector.start(target)

        return ExecutorOutcome(
            status, target.goal, impatience, tuple(completed), preemptions
        )
