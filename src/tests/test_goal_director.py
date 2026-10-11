"""Tests for the GoalDirector lifecycle (S8 pass 2, stages/S8_PASS2_SPEC.md §7).

Shell-level lifecycle: one action in flight (RUNNING does not rebuild the tree),
attempts/cooldown, deadline → preemption, goal change in flight, gate refusal,
periodic re-observation after exhaustion, S6 suppression, and the disabled
default keeping the S7/S8-pass-1 contour identical.
"""

from __future__ import annotations

import types
from pathlib import Path

from src.config import (
    ActuationConfig,
    AutonomyConfig,
    GoalConfig,
    HostConfig,
    MemoryConfig,
    OptionDeclaration,
)
from src.core.actuation import (
    Effect,
    Fact,
    Guard,
    Node,
    NodeKind,
    NodeStatus,
)
from src.host.executor import ExecutorOutcome
from src.host.goal_director import GoalDirector, _first_action
from src.host.loop import build_host_loop
from src.host.sensitivity import DeterministicMeter
from src.tests.ab_environment import (
    A_SERVICED,
    B_SERVICED,
    AbEnvironment,
    ab_affordances,
    ab_outcomes,
)


def _declarations() -> tuple[OptionDeclaration, ...]:
    """Декларации стенда: сервис A/B → соответствующий флаг := 1."""
    return (
        OptionDeclaration("tool:service_a", effect=Effect(A_SERVICED, value=1.0)),
        OptionDeclaration("tool:service_b", effect=Effect(B_SERVICED, value=1.0)),
    )


def _build(
    tmp_path: Path,
    env: AbEnvironment,
    *,
    outcomes=None,
    max_attempts: int = 3,
    deadline_ticks: int = 8,
    cooldown_ticks: int = 1,
    latency_ticks: int = 0,
):
    """Собрать хост с подключённым целевым контуром и стендом A/B."""
    cfg = HostConfig(
        memory=MemoryConfig(enabled=False),
        log_path=str(tmp_path / "t.jsonl"),
        autonomy=AutonomyConfig(enabled=True),
        actuation=ActuationConfig(enabled=True, latency_ticks=latency_ticks),
        goal=GoalConfig(
            enabled=True,
            outcomes=outcomes if outcomes is not None else ab_outcomes(),
            max_attempts=max_attempts,
            deadline_ticks=deadline_ticks,
            cooldown_ticks=cooldown_ticks,
        ),
        option_declarations=_declarations(),
    )
    return build_host_loop(
        cfg,
        meter=DeterministicMeter(),
        probe_fn=env.probe_fn,
        affordances=ab_affordances(),
        environment=env,
    )


class TestWiring:
    """Сборка директора и совместимость выключенного контура."""

    def test_disabled_has_no_director(self, tmp_path: Path) -> None:
        """goal.enabled=False → директора нет, поля телеметрии пусты."""
        cfg = HostConfig(
            memory=MemoryConfig(enabled=False),
            log_path=str(tmp_path / "t.jsonl"),
            actuation=ActuationConfig(enabled=True),
        )
        loop = build_host_loop(cfg, meter=DeterministicMeter())
        try:
            assert loop.goal_director is None
            loop.step_once(0)
            assert loop.last_goal_telemetry == ("", "", 0, False, "", "")
        finally:
            loop.close()

    def test_goal_without_actuation_raises(self, tmp_path: Path) -> None:
        """goal.enabled без actuation.enabled → ValueError (fail-fast)."""
        import pytest

        cfg = HostConfig(
            memory=MemoryConfig(enabled=False),
            log_path=str(tmp_path / "t.jsonl"),
            goal=GoalConfig(enabled=True, outcomes=ab_outcomes()),
        )
        env = AbEnvironment()
        with pytest.raises(ValueError, match="requires actuation.enabled"):
            build_host_loop(cfg, meter=DeterministicMeter(), environment=env)

    def test_goal_without_environment_raises(self, tmp_path: Path) -> None:
        """goal.enabled без environment → ValueError (fail-fast)."""
        import pytest

        cfg = HostConfig(
            memory=MemoryConfig(enabled=False),
            log_path=str(tmp_path / "t.jsonl"),
            actuation=ActuationConfig(enabled=True),
            goal=GoalConfig(enabled=True, outcomes=ab_outcomes()),
        )
        with pytest.raises(ValueError, match="requires an environment"):
            build_host_loop(cfg, meter=DeterministicMeter())


class TestLifecycle:
    """Жизненный цикл: запуск, достижение, повторы, cooldown, исчерпание."""

    def test_goal_achieved_over_ticks(self, tmp_path: Path) -> None:
        """Цель A выбирается, сервис вызван, новое наблюдение подтверждает."""
        env = AbEnvironment()
        loop = _build(tmp_path, env)
        try:
            loop.step_once(0)
            selected, _, _, achieved, status, _ = loop.last_goal_telemetry
            assert selected == "a_serviced"
            assert status == "running"
            assert achieved is False

            loop.step_once(1)
            selected, _, _, achieved, status, _ = loop.last_goal_telemetry
            assert selected == "a_serviced"
            assert status == "success"
            assert achieved is True
            assert env.calls == ["service_a"]
            assert env.observe()["a_serviced"] == 1.0
        finally:
            loop.close()

    def test_only_b_needs_service(self, tmp_path: Path) -> None:
        """A уже обслужен → цель B, вызван service_b (контроль содержательности)."""
        env = AbEnvironment(a_serviced=1.0)
        loop = _build(tmp_path, env)
        try:
            loop.step_once(0)
            loop.step_once(1)
            assert env.calls == ["service_b"]
            assert env.observe()["b_serviced"] == 1.0
        finally:
            loop.close()

    def test_both_serviced_no_calls(self, tmp_path: Path) -> None:
        """Оба обслужены → нет цели, нет вызовов."""
        env = AbEnvironment(a_serviced=1.0, b_serviced=1.0)
        loop = _build(tmp_path, env)
        try:
            for tick in range(3):
                loop.step_once(tick)
            assert env.calls == []
            assert loop.last_goal_telemetry[0] == ""
        finally:
            loop.close()

    def test_running_does_not_restart(self, tmp_path: Path) -> None:
        """Пока RUNNING, дерево не пересобирается: ровно один start."""
        env = AbEnvironment()
        loop = _build(tmp_path, env, latency_ticks=2)
        try:
            for tick in range(2):
                loop.step_once(tick)
            # latency=2: первый тик — start, второй — ещё Running (без повтора).
            assert env.calls == []
            assert loop.last_goal_telemetry[4] == "running"
            for tick in range(2, 5):
                loop.step_once(tick)
            assert env.calls == ["service_a"]
        finally:
            loop.close()

    def test_lie_tool_repeats_bounded(self, tmp_path: Path) -> None:
        """Успех без изменения состояния → повторы ≤ max_attempts."""
        env = AbEnvironment(lie=True)
        loop = _build(tmp_path, env, max_attempts=3, cooldown_ticks=0)
        try:
            for tick in range(12):
                loop.step_once(tick)
            assert env.calls == ["service_a", "service_a", "service_a"]
            assert loop.last_goal_telemetry[2] == 3  # attempts
        finally:
            loop.close()

    def test_forbidden_tool_no_success(self, tmp_path: Path) -> None:
        """Запрещённый инструмент → нет успеха цели, причина в трассе."""
        env = AbEnvironment(forbid=("service_a",))
        loop = _build(tmp_path, env, max_attempts=1)
        try:
            for tick in range(4):
                loop.step_once(tick)
            assert env.observe()["a_serviced"] == 0.0
            assert loop.last_goal_telemetry[3] is False  # achieved
        finally:
            loop.close()

    def test_observation_continues_after_exhaustion(self, tmp_path: Path) -> None:
        """После исчерпания попыток среда переспрашивается (изменение видно)."""
        env = AbEnvironment(lie=True)
        loop = _build(tmp_path, env, max_attempts=1, cooldown_ticks=0)
        try:
            for tick in range(3):
                loop.step_once(tick)
            assert loop.last_goal_telemetry[3] is False
            # Значимое изменение факта цели (A обслужен извне) → цель сброшена,
            # выбирается B (наблюдение не прекращалось).
            env.set_fact("a_serviced", 1.0)
            before = len(env.calls)
            for tick in range(3, 6):
                loop.step_once(tick)
            assert loop.last_goal_telemetry[0] == "b_serviced"
            assert len(env.calls) > before
        finally:
            loop.close()

    def test_deadline_preempts(self, tmp_path: Path) -> None:
        """Превышение deadline → преемпция → попытка завершается провалом."""
        env = AbEnvironment()
        loop = _build(
            tmp_path,
            env,
            latency_ticks=5,
            deadline_ticks=1,
            max_attempts=1,
            cooldown_ticks=0,
        )
        try:
            loop.step_once(0)  # launch
            loop.step_once(1)  # elapsed=1, not > deadline yet → running
            loop.step_once(2)  # elapsed=2 > 1 → deadline
            assert "deadline" in loop.last_goal_telemetry[1]
        finally:
            loop.close()

    def test_goal_change_in_flight(self, tmp_path: Path) -> None:
        """Оператор снял исход A → следующий тик выбирает B (преемпция)."""
        env = AbEnvironment()
        outcomes = ab_outcomes()
        loop = _build(tmp_path, env, outcomes=outcomes, latency_ticks=3)
        try:
            loop.step_once(0)  # launch service_a (running)
            assert loop.last_goal_telemetry[0] == "a_serviced"
            # Оператор изменил предпочтения: только B неудовлетворён.
            loop.goal_director.outcomes = (outcomes[1],)
            loop.step_once(1)
            assert loop.last_goal_telemetry[0] == "b_serviced"
        finally:
            loop.close()

    def test_goal_reached_in_flight(self, tmp_path: Path) -> None:
        """Цель достигнута до терминального poll → событие «в полёте», не смена.

        Среда достигает цели A (имитируем внешнее изменение, пока вызов ещё
        ``RUNNING``). Директор обязан **разрешить активную попытку**: зафиксировать
        достижение именно A, а не молча заменить её другой целью/``None``.
        """
        env = AbEnvironment(b_serviced=1.0)  # только A требует обслуживания
        loop = _build(tmp_path, env, latency_ticks=5)
        try:
            loop.step_once(0)  # launch service_a (running)
            assert loop.last_goal_telemetry[0] == "a_serviced"
            assert loop.last_goal_telemetry[4] == "running"
            # Среда достигла A, пока вызов ещё не терминален.
            env.set_fact("a_serviced", 1.0)
            loop.step_once(1)
            selected, reason, _, achieved, _, _ = loop.last_goal_telemetry
            assert selected == "a_serviced"  # достижение старой цели, не смена
            assert achieved is True
            assert "in flight" in reason
        finally:
            loop.close()

    def test_goal_change_stops_external_operation(self, tmp_path: Path) -> None:
        """Смена цели преемптит внешнюю операцию (не только внутренний reset).

        Долгий вызов A (``latency_ticks`` велик, работа ещё не выполнена) сменяется
        на B: преемпция через executor должна **остановить** эффектор A — работа A
        не выполняется вовсе (``service_a`` не в ``calls``), запускается B.
        """
        env = AbEnvironment()
        outcomes = ab_outcomes()
        loop = _build(tmp_path, env, outcomes=outcomes, latency_ticks=6)
        try:
            loop.step_once(0)  # launch service_a (running, работа не завершена)
            assert loop.last_goal_telemetry[0] == "a_serviced"
            # Оператор снял исход A → преемпция: A не должен выполниться.
            loop.goal_director.outcomes = (outcomes[1],)
            for tick in range(1, 12):
                loop.step_once(tick)
            assert "service_a" not in env.calls  # внешняя операция остановлена
            assert env.calls == ["service_b"]
        finally:
            loop.close()


class TestCoexistence:
    """S6/S8: explore подавляется при активном целевом контуре (§8)."""

    def test_explore_suppressed_while_active(self, tmp_path: Path) -> None:
        """При наличии цели explore() возвращает None."""
        env = AbEnvironment()
        loop = _build(tmp_path, env)
        try:
            loop.step_once(0)
            assert loop.goal_director is not None
            assert loop.goal_director.active is True
            assert loop.explore() is None
        finally:
            loop.close()

    def test_explore_allowed_when_idle(self, tmp_path: Path) -> None:
        """Все исходы удовлетворены → директор неактивен, explore разрешён."""
        env = AbEnvironment(a_serviced=1.0, b_serviced=1.0)
        loop = _build(tmp_path, env)
        try:
            loop.step_once(0)
            assert loop.goal_director is not None
            assert loop.goal_director.active is False
        finally:
            loop.close()


def _action(name: str) -> Node:
    """Action-лист для теста структуры дерева."""
    from src.core.actuation import Actuation, ActuationKind

    return Node(
        NodeKind.ACTION,
        name=name,
        actuation=Actuation(ActuationKind.INVOKE_TOOL, name, name),
    )


def _condition(fact: Fact, value: float) -> Node:
    """Condition-лист для теста структуры дерева."""
    return Node(NodeKind.CONDITION, name=fact.name, guard=Guard(fact, value))


class TestLaunchTrace:
    """Трасса запуска привязана к фактическому листу дерева (§7, п.4)."""

    def test_first_action_finds_leaf_after_subgoal(self) -> None:
        """Sequence(subgoal, action) → первый Action = исполняемый лист."""
        subgoal = _condition(A_SERVICED, 1.0)
        leaf = _action("tool:service_a")
        tree = Node(NodeKind.SEQUENCE, name="tool:service_a", children=(subgoal, leaf))
        found = _first_action(tree)
        assert found is not None
        assert found.name == "tool:service_a"

    def test_first_action_none_for_pure_condition(self) -> None:
        """Чистый Condition (безопасный отказ) → нет исполняемого листа."""
        assert _first_action(_condition(A_SERVICED, 1.0)) is None

    def test_launched_goal_matches_chain_leaf(self, tmp_path: Path) -> None:
        """`launched_goal` — реальный лист цепочки, не победитель скоринга."""
        env = AbEnvironment()
        loop = _build(tmp_path, env)
        try:
            loop.step_once(0)
            attempt = loop.goal_director.tick(0)  # повторный тик для доступа
            assert attempt.chain_goal == "tool:service_a"
            assert attempt.launched_goal == "tool:service_a"
            assert "launched via tool:service_a" in loop.last_goal_telemetry[1]
        finally:
            loop.close()

    def test_launched_goal_is_leaf_not_chain_root(self) -> None:
        """Фактически запущенный лист ≠ корень цепочки при подцели (guard).

        ``service_a`` требует предпосылку ``B обслужен`` (``guard``). Цепочка по
        цели A становится ``Sequence(service_b, service_a)``: корень цепочки —
        ``tool:service_a``, но **фактически запускается** первый лист —
        ``tool:service_b``. Трасса запуска берёт лист, а не корень.
        """
        env = AbEnvironment()

        class _StubLoop:
            probe_effector = types.SimpleNamespace(affordances=ab_affordances())
            executor = None

            def active_task(self) -> str:
                return ""

            def tick_actuation(self, root, facts=None):
                return ExecutorOutcome(NodeStatus.RUNNING, "tool:service_b", 0.0, ())

        declarations = (
            OptionDeclaration(
                "tool:service_a",
                effect=Effect(A_SERVICED, value=1.0),
                guard=Guard(B_SERVICED, 1.0),  # предпосылка: B обслужен
            ),
            OptionDeclaration("tool:service_b", effect=Effect(B_SERVICED, value=1.0)),
        )
        director = GoalDirector(
            loop=_StubLoop(),  # type: ignore[arg-type]
            outcomes=(ab_outcomes()[0],),  # только A
            declarations=declarations,
            environment=env,
            max_depth=3,
        )
        attempt = director.tick(0)
        assert attempt.chain_goal == "tool:service_a"  # корень выведенной цепочки
        assert attempt.launched_goal == "tool:service_b"  # фактический лист
        assert attempt.reason == "launched via tool:service_b"
