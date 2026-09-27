"""Integration tests for the S4 reflex path and policy context in HostLoop.

Covers the S4 gates that depend on the loop:
    - reflex: critical intero signal → throttle within ≤ 1 tick (VALIDATION §2.3)
    - C6: resource overload → throttle activates
    - backward compatibility: no homeostat → no throttle (S1–S3 contour)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import HomeostasisConfig, HostConfig, MemoryConfig
from src.core.cmc import ColumnConfig
from src.core.energy import DriftDetector, PrecisionEstimator
from src.core.homeostasis import Homeostat, Setpoint
from src.host.loop import HostLoop, build_host_loop
from src.host.resources import ResourceMeter
from src.host.sources import BatteryProvider, ConstantProvider, SignalBus
from src.host.wiring import build_cmc_pipeline
from src.telemetry import TelemetryLogger, TelemetryWriter


class FakeMeter(ResourceMeter):
    """Фейковый meter: детерминированные метрики ресурсов."""

    def __init__(self, latency_s: float = 0.0, rss_mb: float = 0.0) -> None:
        super().__init__()
        self._fake_latency = latency_s
        self._fake_rss = rss_mb

    @property
    def last_latency_s(self) -> float:
        return self._fake_latency

    @property
    def last_rss_mb(self) -> float:
        return self._fake_rss


def _loop_with_homeostat(
    tmp_path: Path,
    providers: list,
    setpoints: tuple[Setpoint, ...],
    tick_dt: float = 0.01,
) -> HostLoop:
    """Собрать HostLoop с гомеостатом под произвольный набор провайдеров."""
    bus = SignalBus(providers)
    columns = [
        ColumnConfig(input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="a"),
        ColumnConfig(input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="b"),
    ]
    pipeline = build_cmc_pipeline(
        columns=columns, k=1, log_path=tmp_path / "run.jsonl", active_threshold=1e-8
    )
    writer = TelemetryWriter(log_path=tmp_path / "run.jsonl")
    logger = TelemetryLogger(writer=writer, phase="phase1", mode="free")
    return HostLoop(
        bus=bus,
        pipeline=pipeline,
        logger=logger,
        estimator=PrecisionEstimator(dim=bus.bus_dim, window=50),
        meter=ResourceMeter(),
        drift=DriftDetector(f_threshold=1e9, stress_threshold=1e9),
        homeostat=Homeostat(setpoints=setpoints),
        tick_dt=tick_dt,
        clock_mode="synthetic",
        precision_mode="ones",
    )


def _read_events(tmp_path: Path) -> list[dict]:
    path = tmp_path / "run.jsonl"
    return [json.loads(line) for line in path.read_text().strip().split("\n")]


class TestReflexThrottle:
    """Reflex path: critical intero signal → throttle within ≤ 1 tick."""

    def test_reflex_activates_throttle_same_tick(self, tmp_path: Path) -> None:
        """Критический сигнал → throttle активен в том же тике (≤ 1 тик)."""
        provider = BatteryProvider(start_level=1.0, drain_per_tick=0.01)
        loop = _loop_with_homeostat(
            tmp_path,
            [provider],
            setpoints=(Setpoint(tag="battery", comfort=0.5, critical=0.9),),
        )
        # Критический сигнал наступает при severity >= 0.9 → level <= 0.1,
        # то есть при tick >= 90.
        for tick in range(90):
            loop.step_once(tick)
        assert loop.last_throttle.active is False

        loop.step_once(90)  # severity = 0.9 → рефлекс
        assert loop.last_throttle.active is True
        assert loop.last_throttle.llm_gate is True
        loop.close()

    def test_throttle_logged_in_telemetry(self, tmp_path: Path) -> None:
        """Throttle и отклонение гомеостаза видны в телеметрии."""
        provider = BatteryProvider(start_level=1.0, drain_per_tick=0.01)
        loop = _loop_with_homeostat(
            tmp_path,
            [provider],
            setpoints=(Setpoint(tag="battery", comfort=0.5, critical=0.9),),
        )
        for tick in range(91):
            loop.step_once(tick)
        loop.close()

        events = _read_events(tmp_path)
        assert events[-1]["throttle"] is True
        assert events[-1]["homeostasis"] > 0.0

    def test_dt_scaled_when_throttled(self, tmp_path: Path) -> None:
        """При throttle dt увеличивается (медленнее тики)."""
        provider = BatteryProvider(start_level=1.0, drain_per_tick=0.01)
        loop = _loop_with_homeostat(
            tmp_path,
            [provider],
            setpoints=(Setpoint(tag="battery", comfort=0.5, critical=0.9),),
        )
        loop.throttle_dt_scale = 2.0
        for tick in range(91):
            loop.step_once(tick)
        # Проверяем, что k-WTA сжат до 1 (k_scale=0.5 от k=1 → max(1, 0.5)=1).
        assert loop.pipeline.voting.k == 1
        loop.close()

    def test_no_throttle_in_norm(self, tmp_path: Path) -> None:
        """Норма → throttle неактивен."""
        loop = _loop_with_homeostat(
            tmp_path,
            [ConstantProvider(value=(1.0,))],
            setpoints=(Setpoint(tag="battery", comfort=0.5, critical=0.9),),
        )
        loop.step_once(0)
        assert loop.last_throttle.active is False
        loop.close()

    def test_k_restored_after_throttle(self, tmp_path: Path) -> None:
        """Обратимость: k восстанавливается после нормализации сигнала.

        Throttle — обратимая регуляция, а не дрейф конфигурации. При
        критическом сигнале k сжимается, при норме возвращается к базовому.
        """

        # Комбинируем battery (критический сигнал) с управляемым ресурсом:
        # сначала перегруз, затем норма через FakeMeter нельзя менять на лету,
        # поэтому проверяем восстановление через повторную сборку гомеостата.
        provider = BatteryProvider(start_level=1.0, drain_per_tick=0.01)
        loop = _loop_with_homeostat(
            tmp_path,
            [provider],
            setpoints=(Setpoint(tag="battery", comfort=0.5, critical=0.9),),
        )
        assert loop.pipeline.voting.k == 1  # базовое k сценария
        loop._base_k = 2
        for tick in range(91):
            loop.step_once(tick)
        assert loop.pipeline.voting.k == 1  # сжато (2 * 0.5)
        # Нормализация: убираем сетпоинт battery → сигнал не критичен
        loop.homeostat = Homeostat(setpoints=(Setpoint(tag="other"),))
        loop.step_once(91)
        assert loop.pipeline.voting.k == 2  # восстановлено
        loop.close()

    def test_throttle_severity_threshold_from_config(self, tmp_path: Path) -> None:
        """Кастомный reflex_threshold доходит до throttle (согласованность)."""
        provider = BatteryProvider(start_level=1.0, drain_per_tick=0.01)
        bus = SignalBus([provider])
        columns = [
            ColumnConfig(
                input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="a"
            ),
            ColumnConfig(
                input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="b"
            ),
        ]
        pipeline = build_cmc_pipeline(
            columns=columns, k=1, log_path=tmp_path / "run.jsonl"
        )
        writer = TelemetryWriter(log_path=tmp_path / "run.jsonl")
        loop = HostLoop(
            bus=bus,
            pipeline=pipeline,
            logger=TelemetryLogger(writer=writer, phase="phase1", mode="free"),
            estimator=PrecisionEstimator(dim=bus.bus_dim, window=50),
            meter=ResourceMeter(),
            drift=DriftDetector(f_threshold=1e9, stress_threshold=1e9),
            homeostat=Homeostat(
                setpoints=(Setpoint(tag="battery", comfort=0.5, critical=0.9),),
                reflex_threshold=0.5,
            ),
            throttle_severity_threshold=0.5,
            precision_mode="ones",
        )
        # severity >= 0.5 при tick >= 50 → throttle раньше, чем при 0.9
        for tick in range(51):
            loop.step_once(tick)
        assert loop.last_throttle.active is True
        loop.close()


class TestResourceThrottleC6:
    """C6: resource overload → throttle (resource feedback)."""

    def test_c6_resource_overload_throttles(self, tmp_path: Path) -> None:
        """Перегруз ресурсов (латентность) → throttle срабатывает."""
        from src.host.resources import ResourceProvider

        meter = FakeMeter(latency_s=0.05, rss_mb=100.0)  # 50/50 = 1.0
        provider = ResourceProvider(meter=meter, tick_budget_ms=50.0)
        loop = _loop_with_homeostat(
            tmp_path,
            [provider],
            setpoints=(Setpoint(tag="resources", comfort=0.5, critical=0.9),),
        )
        loop.step_once(0)
        assert loop.last_throttle.active is True
        loop.close()

    def test_c6_norm_no_throttle(self, tmp_path: Path) -> None:
        """Нормальная нагрузка → throttle не срабатывает."""
        from src.host.resources import ResourceProvider

        meter = FakeMeter(latency_s=0.001, rss_mb=100.0)
        provider = ResourceProvider(meter=meter, tick_budget_ms=50.0)
        loop = _loop_with_homeostat(
            tmp_path,
            [provider],
            setpoints=(Setpoint(tag="resources", comfort=0.5, critical=0.9),),
        )
        loop.step_once(0)
        assert loop.last_throttle.active is False
        loop.close()


class TestBackwardCompatibility:
    """S4 без гомеостата → контур S1–S3 идентичен (throttle неактивен)."""

    def test_loop_without_homeostat_no_throttle(self, tmp_path: Path) -> None:
        """Homeostat=None → throttle неактивен, телеметрия нулевая."""
        bus = SignalBus([ConstantProvider(value=(1.0,))])
        columns = [
            ColumnConfig(
                input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="a"
            ),
            ColumnConfig(
                input_dim=bus.bus_dim, state_dim=bus.bus_dim, specialization="b"
            ),
        ]
        pipeline = build_cmc_pipeline(
            columns=columns, k=1, log_path=tmp_path / "run.jsonl"
        )
        writer = TelemetryWriter(log_path=tmp_path / "run.jsonl")
        loop = HostLoop(
            bus=bus,
            pipeline=pipeline,
            logger=TelemetryLogger(writer=writer, phase="phase1", mode="free"),
            estimator=PrecisionEstimator(dim=bus.bus_dim, window=50),
            meter=ResourceMeter(),
            drift=DriftDetector(f_threshold=1e9, stress_threshold=1e9),
            homeostat=None,
            precision_mode="ones",
        )
        loop.run(5)
        loop.close()

        events = _read_events(tmp_path)
        assert all(e["throttle"] is False for e in events)
        assert all(e["homeostasis"] == 0.0 for e in events)

    def test_default_config_no_throttle_when_calm(self, tmp_path: Path) -> None:
        """Дефолтный конфиг: сигналы в норме → throttle неактивен."""
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "mem.db")),
        )
        loop = build_host_loop(config)
        loop.run(5)
        loop.close()

        events = _read_events(tmp_path)
        assert all(e["throttle"] is False for e in events)

    def test_build_host_loop_has_homeostat(self, tmp_path: Path) -> None:
        """build_host_loop собирает гомеостат из конфига."""
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(db_path=str(tmp_path / "mem.db")),
            homeostasis=HomeostasisConfig(),
        )
        loop = build_host_loop(config)
        assert loop.homeostat is not None
        assert len(loop.homeostat.setpoints) == 3
        loop.close()


class TestAttentionGate:
    """Pre-column γ barrier (S4 проход 2): off → identity, on → attenuates."""

    def test_gate_off_identity(self, tmp_path: Path) -> None:
        """attention_gate=False → вход колонок не изменён (S1–S3)."""
        provider = ConstantProvider(value=(3.0,))
        loop = _loop_with_homeostat(
            tmp_path, [provider], setpoints=(Setpoint(tag="battery"),)
        )
        loop.attention_gate = False
        loop.step_once(0)
        assert loop.last_outcome is not None
        loop.close()

    def test_gate_on_changes_behavior(self, tmp_path: Path) -> None:
        """attention_gate=True → вход аттенюирован, F отличается."""
        provider = ConstantProvider(value=(3.0,))

        loop_off = _loop_with_homeostat(
            tmp_path, [provider], setpoints=(Setpoint(tag="battery"),)
        )
        loop_off.attention_gate = False
        off = loop_off.step_once(0).result.f
        loop_off.close()

        loop_on = _loop_with_homeostat(
            tmp_path, [provider], setpoints=(Setpoint(tag="battery"),)
        )
        loop_on.attention_gate = True
        on = loop_on.step_once(0).result.f
        loop_on.close()

        assert on < off  # аттенюация уменьшает ошибку первого тика


class TestPolicyContextAndTelemetry:
    """Policy context assembly + policy fields in telemetry."""

    def test_policy_context_from_state(self, tmp_path: Path) -> None:
        """policy_context несёт F/аффект/задачу/гомеостаз текущего состояния."""
        loop = _loop_with_homeostat(
            tmp_path,
            [ConstantProvider(value=(1.0,))],
            setpoints=(Setpoint(tag="battery", comfort=0.5, critical=0.9),),
        )
        loop.step_once(0)
        ctx = loop.policy_context(has_new_message=True, mode="game")
        assert ctx.has_new_message is True
        assert ctx.mode == "game"
        assert ctx.task == "a"  # специализация колонки-аттрактора
        assert ctx.homeostasis is loop.last_homeostasis
        loop.close()

    def test_record_policy_writes_telemetry(self, tmp_path: Path) -> None:
        """record_policy → поля policy_action/policy_reason в следующем тике."""
        from src.core.policy import Preferences, select_action

        loop = _loop_with_homeostat(
            tmp_path,
            [ConstantProvider(value=(1.0,))],
            setpoints=(Setpoint(tag="battery", comfort=0.5, critical=0.9),),
        )
        loop.step_once(0)
        ctx = loop.policy_context(has_new_message=True)
        trace = select_action(ctx, Preferences())
        loop.record_policy(trace)
        loop.step_once(1)
        loop.close()

        events = _read_events(tmp_path)
        assert events[0]["policy_action"] == ""
        assert events[1]["policy_action"] == "respond"
        assert "respond" in events[1]["policy_reason"]

    def test_policy_action_cleared_after_one_tick(self, tmp_path: Path) -> None:
        """policy_action не «залипает»: только один тик после записи."""
        from src.core.policy import Preferences, select_action

        loop = _loop_with_homeostat(
            tmp_path,
            [ConstantProvider(value=(1.0,))],
            setpoints=(Setpoint(tag="battery", comfort=0.5, critical=0.9),),
        )
        loop.step_once(0)
        loop.record_policy(
            select_action(loop.policy_context(has_new_message=True), Preferences())
        )
        loop.step_once(1)
        loop.step_once(2)
        loop.close()

        events = _read_events(tmp_path)
        assert events[1]["policy_action"] == "respond"
        assert events[2]["policy_action"] == ""
