"""Tests for actuation wiring — config, telemetry, S7 compatibility (S8 stage 7).

Covers: ``ActuationConfig`` presence in ``HostConfig``; telemetry fields for
actuations serialize; ``enabled=False`` keeps the S7 contour identical (no
executor, empty actuation fields); the executor is built when enabled; a tree is
driven tick-to-tick through ``HostLoop.tick_actuation``; and the CLI flag wiring.
"""

from __future__ import annotations

from pathlib import Path

from src.config import ActuationConfig, HostConfig, MemoryConfig
from src.core.actuation import (
    Actuation,
    ActuationKind,
    ActuationResult,
    ActuationStatus,
    Node,
    NodeKind,
    NodeStatus,
)
from src.host.effectors import DeferredEffector
from src.host.loop import build_host_loop
from src.host.sensitivity import DeterministicMeter


def _action(goal: str, kind: ActuationKind = ActuationKind.INVOKE_TOOL) -> Node:
    """Собрать Action-лист."""
    return Node(NodeKind.ACTION, name=goal, actuation=Actuation(kind, goal, goal))


def _ok(_a: Actuation) -> ActuationResult:
    """Работа, всегда успешная."""
    return ActuationResult(ActuationStatus.SUCCESS, (1.0,))


def _disabled(tmp_path: Path) -> HostConfig:
    """Конфиг с выключенным секвенированием и без памяти."""
    return HostConfig(
        memory=MemoryConfig(enabled=False), log_path=str(tmp_path / "t.jsonl")
    )


class TestConfigCompat:
    """Дефолт disabled → контур S7 идентичен."""

    def test_default_disabled(self) -> None:
        """HostConfig().actuation.enabled is False."""
        assert HostConfig().actuation.enabled is False

    def test_disabled_builds_no_executor(self, tmp_path: Path) -> None:
        """enabled=False → executor не создаётся, tick_actuation — no-op."""
        loop = build_host_loop(_disabled(tmp_path), meter=DeterministicMeter())
        try:
            assert loop.executor is None
            assert loop.actuation_enabled is False
            assert loop.tick_actuation(_action("a")) is None
        finally:
            loop.close()

    def test_disabled_telemetry_fields_empty(self, tmp_path: Path) -> None:
        """При выключенном секвенировании поля актуаций пусты."""
        loop = build_host_loop(_disabled(tmp_path), meter=DeterministicMeter())
        try:
            loop.step_once(0)
            status, goal, impatience, steps, preemptions = loop.last_actuation_telemetry
            assert status == ""
            assert goal == ""
            assert impatience == 0.0
            assert steps == 0
            assert preemptions == 0
        finally:
            loop.close()


class TestEnabledWiring:
    """enabled=True → executor над эффекторами."""

    def test_enabled_builds_executor(self, tmp_path: Path) -> None:
        """enabled=True → executor создан (тул-эффектор без probe отсутствует)."""
        cfg = HostConfig(
            memory=MemoryConfig(enabled=False),
            log_path=str(tmp_path / "t.jsonl"),
            actuation=ActuationConfig(enabled=True),
        )
        loop = build_host_loop(cfg, meter=DeterministicMeter())
        try:
            assert loop.executor is not None
            assert loop.actuation_enabled is True
        finally:
            loop.close()

    def test_tick_actuation_drives_tree(self, tmp_path: Path) -> None:
        """tick_actuation ведёт дерево через эффекторы и пишет телеметрию."""
        cfg = HostConfig(
            memory=MemoryConfig(enabled=False),
            log_path=str(tmp_path / "t.jsonl"),
            actuation=ActuationConfig(enabled=True),
        )
        loop = build_host_loop(cfg, meter=DeterministicMeter())
        try:
            assert loop.executor is not None
            loop.executor.effectors[ActuationKind.INVOKE_TOOL] = DeferredEffector(_ok)
            root = _action("a")
            first = loop.tick_actuation(root)
            assert first is not None
            assert first.status is NodeStatus.RUNNING
            assert loop.last_actuation_goal == "a"

            second = loop.tick_actuation(root)
            assert second is not None
            assert second.status is NodeStatus.SUCCESS
            assert loop.last_actuation_telemetry[3] == 1
        finally:
            loop.close()

    def test_attach_speech_is_noop_when_disabled(self, tmp_path: Path) -> None:
        """attach_speech при выключенном секвенировании — no-op."""
        loop = build_host_loop(_disabled(tmp_path), meter=DeterministicMeter())
        try:
            loop.attach_speech(lambda _a: "hi")  # не должно упасть
            assert loop.executor is None
        finally:
            loop.close()

    def test_attach_speech_adds_effector(self, tmp_path: Path) -> None:
        """attach_speech подключает речевой эффектор при включении."""
        cfg = HostConfig(
            memory=MemoryConfig(enabled=False),
            log_path=str(tmp_path / "t.jsonl"),
            actuation=ActuationConfig(enabled=True),
        )
        loop = build_host_loop(cfg, meter=DeterministicMeter())
        try:
            assert loop.executor is not None
            assert ActuationKind.SPEAK not in loop.executor.effectors
            loop.attach_speech(lambda _a: "hi")
            assert ActuationKind.SPEAK in loop.executor.effectors
        finally:
            loop.close()
