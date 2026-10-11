"""End-to-end projection chain: sensorics → activities → task → policy (ADR-0013).

Механизм проекций подтверждён unit-тестами. Здесь — три отдельные вещи:

1. **Конфигурация реально применяется:** у собранного ``build_host_loop``
   ``resolved_reads`` отражает объявленные reads.
2. **Избирательность:** разные возмущения сегментов дают разных победителей
   (по тегу специализации).
3. **Различие меняет решение:** на одной и той же последовательности входов
   reads- и legacy-конфигурации приводят к разному policy-действию —
   прослеживаемо через activities/F.

Контроль: одинаковый alpha, attention_gate выключен, память выключена,
детерминированный synthetic clock.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from src.config import ColumnParams, HostConfig, MemoryConfig
from src.core.cmc import ColumnConfig
from src.core.energy import DriftDetector, PrecisionEstimator
from src.core.policy import select_action
from src.host.loop import HostLoop, build_host_loop
from src.host.resources import ResourceMeter
from src.host.sources import ConstantProvider, SignalBus, StepProvider
from src.host.wiring import build_cmc_pipeline
from src.telemetry import TelemetryLogger, TelemetryWriter

SEGMENTS = {"a": 1, "b": 1}


def _manual_loop(
    tmp_path: Path,
    providers: list,
    columns_spec: Sequence[tuple[str, tuple[str, ...] | None]],
) -> HostLoop:
    """HostLoop с контролируемыми сегментами a/b и явными reads колонок."""
    bus = SignalBus(providers)
    dim = bus.bus_dim
    columns = [
        ColumnConfig(input_dim=dim, state_dim=dim, specialization=tag, reads=reads)
        for tag, reads in columns_spec
    ]
    pipeline = build_cmc_pipeline(
        columns=columns,
        k=1,
        log_path=tmp_path / "run.jsonl",
        active_threshold=1e-8,
        segment_dims=SEGMENTS,
    )
    logger = TelemetryLogger(
        writer=TelemetryWriter(log_path=tmp_path / "run.jsonl"),
        phase="phase1",
        mode="free",
    )
    return HostLoop(
        bus=bus,
        pipeline=pipeline,
        logger=logger,
        estimator=PrecisionEstimator(dim=dim, window=50),
        meter=ResourceMeter(),
        drift=DriftDetector(f_threshold=1e9, stress_threshold=1e9),
        tick_dt=0.01,
        clock_mode="synthetic",
        precision_mode="ones",
    )


def _step_perturbing(tag: str, delta: float, step_at: int = 10) -> list:
    """Провайдеры: скачок delta на сегменте ``tag``, второй сегмент — 0."""
    providers: list = []
    for name in ("a", "b"):
        if name == tag:
            providers.append(
                StepProvider(before=(0.0,), after=(delta,), step_at=step_at, tag=name)
            )
        else:
            providers.append(ConstantProvider(value=(0.0,), tag=name))
    return providers


class TestBuiltHostReads:
    """1. Конфигурация реально применяется к собранному хосту."""

    def test_resolved_reads_reflect_config(self, tmp_path: Path) -> None:
        """build_host_loop: resolved_reads == объявленные reads (не весь вход)."""
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(enabled=False),
            columns=(
                ColumnParams(specialization="battery", reads=("battery",)),
                ColumnParams(specialization="circadian", reads=("circadian",)),
            ),
        )
        loop = build_host_loop(config)
        loop.close()

        assert loop.pipeline.ensemble.resolved_reads == (
            ("battery",),
            ("circadian",),
        )

    def test_default_config_reads_full_input(self, tmp_path: Path) -> None:
        """Дефолт (без reads) → колонки читают весь вход (legacy-совместимость)."""
        config = HostConfig(
            log_path=str(tmp_path / "run.jsonl"),
            memory=MemoryConfig(enabled=False),
        )
        loop = build_host_loop(config)
        loop.close()

        full = tuple(s.name for s in loop.bus.segments)
        assert all(names == full for names in loop.pipeline.ensemble.resolved_reads)


class TestSelectivity:
    """2. Разные возмущения → разные специализации (по тегу)."""

    COLS: ClassVar[Sequence[tuple[str, tuple[str, ...] | None]]] = [
        ("a_only", ("a",)),
        ("b_only", ("b",)),
    ]

    def test_perturb_a_selects_a_only(self, tmp_path: Path) -> None:
        loop = _manual_loop(tmp_path, list(_step_perturbing("a", 5.0)), self.COLS)
        for tick in range(40):
            loop.step_once(tick)
        loop.close()

        assert loop.active_task() == "a_only"

    def test_perturb_b_selects_b_only(self, tmp_path: Path) -> None:
        loop = _manual_loop(tmp_path, list(_step_perturbing("b", 5.0)), self.COLS)
        for tick in range(40):
            loop.step_once(tick)
        loop.close()

        assert loop.active_task() == "b_only"

    def test_legacy_pair_stays_identical(self, tmp_path: Path) -> None:
        """reads=None: обе колонки читают весь вход → активности идентичны."""
        loop = _manual_loop(
            tmp_path, list(_step_perturbing("a", 5.0)), [("c0", None), ("c1", None)]
        )
        for tick in range(40):
            loop.step_once(tick)
        outcome = loop.last_outcome
        loop.close()

        assert outcome is not None
        assert outcome.activities is not None
        assert outcome.activities[0] == pytest.approx(outcome.activities[1])


class TestDecisionImpact:
    """3. Различие проекций меняет policy-решение (прослеживаемо)."""

    def _final_decision(
        self,
        tmp_path: Path,
        columns_spec: list[tuple[str, tuple[str, ...] | None]],
        delta: float,
        ticks: int = 11,
    ) -> tuple[str, float, str]:
        loop = _manual_loop(tmp_path, list(_step_perturbing("a", delta)), columns_spec)
        for tick in range(ticks):
            outcome = loop.step_once(tick)
        context = loop.policy_context(has_new_message=False)
        trace = select_action(context, HostConfig().policy.preferences)
        loop.close()
        return trace.chosen.value, outcome.result.f, loop.active_task()

    def test_reads_vs_legacy_differ_in_decision(self, tmp_path: Path) -> None:
        """Один вход, delta=1.2: legacy → initiative, reads → silent.

        Legacy дублирует ошибку возмущённого сегмента на обе колонки (FFF выше
        при том же входе) → переходит порог инициативы; reads — нет.
        """
        reads_choice, reads_f, _ = self._final_decision(
            tmp_path / "reads", [("a_only", ("a",)), ("b_only", ("b",))], delta=1.2
        )
        legacy_choice, legacy_f, _ = self._final_decision(
            tmp_path / "legacy", [("c0", None), ("c1", None)], delta=1.2
        )

        assert reads_f < legacy_f  # разная обработка → разный F
        assert reads_choice == "silent"
        assert legacy_choice == "initiative"
        assert reads_choice != legacy_choice  # различие доходит до решения

    def test_task_reflects_perturbed_segment(self, tmp_path: Path) -> None:
        """Аттрактор выбирает читающую возмущённый сегмент колонку."""
        _, _, task = self._final_decision(
            tmp_path, [("a_only", ("a",)), ("b_only", ("b",))], delta=5.0
        )

        assert task == "a_only"

    def test_context_task_not_used_by_policy(self) -> None:
        """Наблюдение: policy-оценки не читают context.task.

        Смена task при равных F/stress/гомеостазе не меняет решение —
        значит различие специализаций доходит до policy только через
        агрегат F, а не через идентичность задачи (текущее поведение).
        """
        from src.core.homeostasis import HomeostasisState
        from src.core.policy import PolicyContext

        prefs = HostConfig().policy.preferences
        homeo = HomeostasisState(
            signals=(), max_deviation=0.0, severity=0.0, is_critical=False
        )

        def _context(task: str) -> PolicyContext:
            return PolicyContext(
                f=0.5,
                valence=0.0,
                stress=0.0,
                task=task,
                homeostasis=homeo,
                has_new_message=False,
            )

        trace_a = select_action(_context("a_only"), prefs)
        trace_b = select_action(_context("b_only"), prefs)

        assert trace_a.chosen == trace_b.chosen
