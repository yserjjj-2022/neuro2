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
from src.host.gate import CapabilityGate
from src.host.loop import HostLoop, build_host_loop
from src.host.sensitivity import DeterministicMeter
from src.tests.ab_environment import (
    A_SERVICED,
    B_SERVICED,
    AbEnvironment,
    ab_affordances,
    ab_gate_denying,
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
    gate: CapabilityGate | None = None,
    **goal_kwargs,
):
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
        gate=gate,
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

    def test_gate_deny_no_call(self, tmp_path: Path) -> None:
        """Gate deny → запрос не доходит до сервиса (calls пуст), причина в трассе.

        Это проверка **авторизации**: право ``ACT_REVERSIBLE`` отозвано, поэтому
        ``ProbeEffector`` отклоняет запрос до транспорта. Раньше это называлось
        «запретом действия», но фактически проверяло отказ сервиса. Причина отказа
        (``denied ...``) доходит до ``goal_reason``.
        """
        env = AbEnvironment()
        loop = _build(tmp_path, env, gate=ab_gate_denying("service_a"))
        try:
            reasons = [self._step(loop, tick)[1] for tick in range(5)]
            assert env.calls == []  # deny до вызова — сервис не тронут
            assert env.observe()["a_serviced"] == 0.0
            assert loop.last_goal_telemetry[3] is False
            assert any("denied" in reason for reason in reasons)  # авторизация
        finally:
            loop.close()

    def test_service_refusal_call_recorded(self, tmp_path: Path) -> None:
        """Сервис отклонил запрос → вызов есть, состояние не изменилось.

        Отличие от Gate deny: запрос **дошёл** до транспорта (``calls`` содержит
        сервис), но сервис ответил ошибкой → состояние не изменилось, цель
        остаётся открытой. Причина (``probe error``) доходит до ``goal_reason`` —
        атрибуция отказа отличается от Gate deny.
        """
        env = AbEnvironment(forbid=("service_a",))
        loop = _build(tmp_path, env, max_attempts=1)
        try:
            reasons = [self._step(loop, tick)[1] for tick in range(5)]
            assert env.calls == ["service_a"]  # вызов состоялся
            assert env.observe()["a_serviced"] == 0.0
            assert loop.last_goal_telemetry[3] is False
            assert any("probe error" in reason for reason in reasons)  # сервис
        finally:
            loop.close()

    @staticmethod
    def _step(loop: HostLoop, tick: int) -> tuple[str, str, int, bool, str, str]:
        """Сделать тик и вернуть телеметрию цели (для сбора причин по тикам)."""
        loop.step_once(tick)
        return loop.last_goal_telemetry

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
        """Симметричные A- и B-ситуации → разные инструменты при равном F.

        Ситуации **симметричны**: в одной требует обслуживания только A, в другой
        — только B (не «оба» против «уже обслужен A»). Общее возбуждение
        сопоставимо (те же провайдеры/seed; различие — какой исход
        неудовлетворён). Проверяем, что различие действия — содержательное, а не
        следствие смены готовности действовать: **численные метрики F на тике
        выбора совпадают**, а выбранные инструменты различны.
        """
        env_a = AbEnvironment(b_serviced=1.0)  # только A требует обслуживания
        loop_a = _build(tmp_path / "a", env_a)
        try:
            loop_a.step_once(0)  # тик выбора: цель выбрана, запущен service_a
            f_a = loop_a.last_free_energy
            goal_a = loop_a.last_goal_telemetry[0]
            loop_a.step_once(1)  # завершение вызова
        finally:
            loop_a.close()

        env_b = AbEnvironment(a_serviced=1.0)  # только B требует обслуживания
        loop_b = _build(tmp_path / "b", env_b)
        try:
            loop_b.step_once(0)  # тик выбора: цель выбрана, запущен service_b
            f_b = loop_b.last_free_energy
            goal_b = loop_b.last_goal_telemetry[0]
            loop_b.step_once(1)  # завершение вызова
        finally:
            loop_b.close()

        # Содержательный выбор: разные цели и разные инструменты.
        assert goal_a == "a_serviced"
        assert goal_b == "b_serviced"
        assert env_a.calls == ["service_a"]
        assert env_b.calls == ["service_b"]
        # Возбуждение сопоставимо на **тике выбора** (не «сработал другой порог
        # готовности»): F зафиксирована сразу после step_once(0).
        assert abs(f_a - f_b) < 1e-6, (f_a, f_b)

    def test_determinism(self, tmp_path: Path) -> None:
        """Одинаковый вход → одинаковая **полная** трасса (по тикам + вызовы).

        Сравнивается вся последовательность решений (не только последний тик):
        иначе разные промежуточные решения с одинаковым финалом прошли бы мимо.
        """
        traces = []
        for run in range(2):
            env = AbEnvironment()
            loop = _build(tmp_path / f"run{run}", env)
            try:
                per_tick = []
                for tick in range(4):
                    loop.step_once(tick)
                    per_tick.append(loop.last_goal_telemetry)
                traces.append((tuple(per_tick), tuple(env.calls)))
            finally:
                loop.close()
        assert traces[0] == traces[1]
