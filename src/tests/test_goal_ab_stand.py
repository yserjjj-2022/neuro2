"""End-to-end acceptance tests on the assembled host (S8 pass 2, SPEC §13).

Runs through the **real control loop** (``HostLoop.step_once`` with the
``GoalDirector`` wired in) — not by hand-calling ``select_option``/
``tick_actuation``. Covers the §13 scenarios and, crucially, the **F-control**:
two situations with comparable general excitation ("A unserved" vs "B unserved")
lead to **different** instrumental actions (``service_a`` vs ``service_b``) —
otherwise only readiness-to-act is shown, not a contentful choice.
"""

from __future__ import annotations

from pathlib import Path

from src.config import (
    ActuationConfig,
    AutonomyConfig,
    GoalConfig,
    HostConfig,
    MemoryConfig,
    OptionDeclaration,
)
from src.core.actuation import Effect
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


def _build(tmp_path: Path, env: AbEnvironment, **goal_kwargs):
    """Собрать хост через рабочий цикл с целевым контуром."""
    cfg = HostConfig(
        memory=MemoryConfig(enabled=False),
        log_path=str(tmp_path / "t.jsonl"),
        autonomy=AutonomyConfig(enabled=True),
        actuation=ActuationConfig(enabled=True),
        goal=GoalConfig(enabled=True, outcomes=ab_outcomes(), **goal_kwargs),
        option_declarations=_declarations(),
    )
    return build_host_loop(
        cfg,
        meter=DeterministicMeter(),
        probe_fn=env.probe_fn,
        affordances=ab_affordances(),
        environment=env,
    )


class TestAcceptanceScenarios:
    """Сценарии §13 SPEC через рабочий цикл."""

    def test_only_a_needs_service(self, tmp_path: Path) -> None:
        """Только A требует обслуживания → выбран service_a, подтверждено."""
        env = AbEnvironment(b_serviced=1.0)
        loop = _build(tmp_path, env)
        try:
            loop.run(4)
            assert env.calls == ["service_a"]
            assert env.observe()["a_serviced"] == 1.0
            assert loop.last_goal_telemetry[3] is True
        finally:
            loop.close()

    def test_only_b_needs_service(self, tmp_path: Path) -> None:
        """Только B требует обслуживания → выбран service_b."""
        env = AbEnvironment(a_serviced=1.0)
        loop = _build(tmp_path, env)
        try:
            loop.run(4)
            assert env.calls == ["service_b"]
            assert env.observe()["b_serviced"] == 1.0
        finally:
            loop.close()

    def test_both_unserved_services_both(self, tmp_path: Path) -> None:
        """Оба не обслужены → обслуживаются оба (по приоритету), затем цель=None."""
        env = AbEnvironment()
        loop = _build(tmp_path, env)
        try:
            loop.run(6)
            assert env.calls == ["service_a", "service_b"]
            assert loop.last_goal_telemetry[0] == ""
        finally:
            loop.close()

    def test_both_serviced_no_calls(self, tmp_path: Path) -> None:
        """Оба обслужены → нет ненужных вызовов (selected_goal=None)."""
        env = AbEnvironment(a_serviced=1.0, b_serviced=1.0)
        loop = _build(tmp_path, env)
        try:
            loop.run(5)
            assert env.calls == []
            assert loop.last_goal_telemetry[0] == ""
        finally:
            loop.close()

    def test_forbidden_action_traced(self, tmp_path: Path) -> None:
        """Действие запрещено → нет изменения состояния, причина в трассе."""
        env = AbEnvironment(forbid=("service_a",))
        loop = _build(tmp_path, env, max_attempts=1)
        try:
            loop.run(5)
            assert env.observe()["a_serviced"] == 0.0
            assert loop.last_goal_telemetry[3] is False
        finally:
            loop.close()

    def test_success_without_state_change_bounded(self, tmp_path: Path) -> None:
        """Успех вызова без изменения состояния → цель не закрыта, ≤ max_attempts."""
        env = AbEnvironment(lie=True)
        loop = _build(tmp_path, env, max_attempts=2, cooldown_ticks=0)
        try:
            loop.run(10)
            assert env.calls == ["service_a", "service_a"]
            assert loop.last_goal_telemetry[3] is False
        finally:
            loop.close()

    def test_operator_changes_preference(self, tmp_path: Path) -> None:
        """Оператор изменил предпочтение → следующий выбор отражает новую цель."""
        env = AbEnvironment()
        outcomes = ab_outcomes()
        loop = _build(tmp_path, env)
        try:
            loop.step_once(0)
            assert loop.last_goal_telemetry[0] == "a_serviced"
            loop.step_once(1)
            # Оператор снял исход A (уже не предпочтителен).
            loop.goal_director.outcomes = (outcomes[1],)
            loop.step_once(2)
            assert loop.last_goal_telemetry[0] == "b_serviced"
        finally:
            loop.close()


class TestFControl:
    """Контроль против эффекта F (§13): одинаковое возбуждение → разный инструмент."""

    def test_comparable_excitation_different_tool(self, tmp_path: Path) -> None:
        """A- и B-ситуации → разные инструменты (service_a vs service_b).

        Общее возбуждение сопоставимо (те же провайдеры/seed, отличаются лишь
        предпочтительный исход и факт среды); различие действия — содержательное,
        не следствие смены готовности действовать.
        """
        env_a = AbEnvironment()
        loop_a = _build(tmp_path / "a", env_a)
        try:
            loop_a.run(3)
        finally:
            loop_a.close()

        env_b = AbEnvironment(a_serviced=1.0)
        loop_b = _build(tmp_path / "b", env_b)
        try:
            loop_b.run(3)
        finally:
            loop_b.close()

        assert env_a.calls == ["service_a"]
        assert env_b.calls == ["service_b"]
        assert env_a.calls != env_b.calls

    def test_determinism(self, tmp_path: Path) -> None:
        """Одинаковый вход → одинаковая трасса выбора."""
        results = []
        for run in range(2):
            env = AbEnvironment()
            loop = _build(tmp_path / f"run{run}", env)
            try:
                for tick in range(4):
                    loop.step_once(tick)
                results.append(loop.last_goal_telemetry)
            finally:
                loop.close()
        assert results[0] == results[1]
