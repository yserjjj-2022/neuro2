"""Effectors — Imperative Shell for actuation side effects (S8 stage 6).

One contract for both speech and tool invocation (ROS Action Server idiom,
ADR-0012 §7): ``Running``/``Success``/``Failure``/``Preempted``. An effector
never blocks the tick: ``start`` kicks off work, ``poll`` reports the current
status, ``preempt`` aborts. Work is deferred by a deterministic number of ticks
(``latency_ticks``) so runs are reproducible (analogous to ``DeterministicMeter``).

Tool invocation always goes through the capability gate (``ProbeEffector``,
ADR-0005 §9): a denied or failing tool yields ``Failure`` — never a silent
allow. Errors never crash the tick.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from src.core.actuation import (
    Actuation,
    ActuationResult,
    ActuationStatus,
)
from src.host.probe import ProbeEffector
from src.mcp.probe import ProbeRequest

logger = logging.getLogger(__name__)

# Синхронная работа активации: актуация → результат (за гейтом, если нужно).
WorkFn = Callable[[Actuation], ActuationResult]


@runtime_checkable
class Effector(Protocol):
    """Контракт эффектора: не блокирует тик (S8 этап 6).

    Attributes:
        goal: Идентификатор исполняемой активации или None (простой).
    """

    @property
    def goal(self) -> str | None:
        """Что сейчас исполняется (None — эффектор свободен)."""
        ...

    def start(self, actuation: Actuation) -> None:
        """Начать активацию (работа откладывается, тик не блокируется)."""
        ...

    def poll(self) -> ActuationResult:
        """Текущий статус активации (``Running`` до готовности)."""
        ...

    def preempt(self) -> None:
        """Прервать активацию: статус → ``Preempted``."""
        ...


class DeferredEffector:
    """Эффектор с отложенной синхронной работой (детерминизм, S8 этап 6).

    ``start`` фиксирует активацию и счётчик тиков; ``poll`` возвращает
    ``Running``, пока не истёк ``latency_ticks``, затем выполняет работу **один
    раз** и кэширует результат. ``preempt`` переводит в ``Preempted``.

    Attributes:
        work: Синхронная работа активации.
        latency_ticks: Через сколько тиков работа завершается (>= 0).
    """

    def __init__(self, work: WorkFn, *, latency_ticks: int = 0) -> None:
        """Создать эффектор.

        Args:
            work: Работа активации (вызывается один раз при готовности).
            latency_ticks: Задержка завершения в тиках (0 — на первом poll).

        Raises:
            ValueError: Если ``latency_ticks < 0``.
        """
        if latency_ticks < 0:
            raise ValueError(f"latency_ticks must be >= 0, got {latency_ticks}")
        self.work = work
        self.latency_ticks = latency_ticks
        self._actuation: Actuation | None = None
        self._elapsed = 0
        self._result: ActuationResult | None = None

    @property
    def goal(self) -> str | None:
        """Идентификатор исполняемой активации или None (простой)."""
        return self._actuation.goal if self._actuation is not None else None

    def start(self, actuation: Actuation) -> None:
        """Начать активацию (сбросить прежнее состояние).

        Args:
            actuation: Активация для исполнения.
        """
        self._actuation = actuation
        self._elapsed = 0
        self._result = None

    def poll(self) -> ActuationResult:
        """Текущий статус: ``Running`` до готовности, затем результат (кэш)."""
        if self._result is not None:
            return self._result
        if self._actuation is None:
            return ActuationResult(ActuationStatus.FAILURE)
        if self._elapsed < self.latency_ticks:
            self._elapsed += 1
            return ActuationResult(ActuationStatus.RUNNING)
        try:
            self._result = self.work(self._actuation)
        except Exception as exc:  # noqa: BLE001 — сбой эффектора не роняет тик
            logger.error("effector failed: %s (%s)", self._actuation.goal, exc)
            self._result = ActuationResult(ActuationStatus.FAILURE)
        return self._result

    def preempt(self) -> None:
        """Прервать активацию: статус → ``Preempted`` (активация сброшена)."""
        self._actuation = None
        self._elapsed = 0
        self._result = ActuationResult(ActuationStatus.PREEMPTED)


class ToolEffector(DeferredEffector):
    """Эффектор вызова MCP-тула через гейт (S8 этап 6).

    Работа делегируется ``ProbeEffector`` (единая точка side-effect, ADR-0005
    §9): отказ гейта или сбой транспорта → ``Failure`` (никогда не роняет тик).
    """

    def __init__(self, probe: ProbeEffector, *, latency_ticks: int = 0) -> None:
        """Создать эффектор тула.

        Args:
            probe: Зондирующий эффектор (транспорт + гейт).
            latency_ticks: Задержка завершения в тиках.
        """
        self.probe = probe
        super().__init__(self._invoke, latency_ticks=latency_ticks)

    def _invoke(self, actuation: Actuation) -> ActuationResult:
        """Вызвать тул через гейт и перевести результат в контракт актуации."""
        result = self.probe.probe(
            ProbeRequest(affordance=actuation.payload, reason=actuation.goal)
        )
        status = ActuationStatus.SUCCESS if result.success else ActuationStatus.FAILURE
        return ActuationResult(status, result.data)


class SpeechEffector(DeferredEffector):
    """Эффектор речи (S8 этап 6).

    Работа — инъецированная функция речи (обёртка ``SpeechController``);
    ``None`` (не говорим / сбой LLM) → ``Failure``.
    """

    def __init__(
        self, speak: Callable[[Actuation], str | None], *, latency_ticks: int = 0
    ) -> None:
        """Создать речевой эффектор.

        Args:
            speak: Функция генерации речи (актуация → текст или None).
            latency_ticks: Задержка завершения в тиках.
        """
        self.speak = speak
        super().__init__(self._speak, latency_ticks=latency_ticks)

    def _speak(self, actuation: Actuation) -> ActuationResult:
        """Сгенерировать речь; None → ``Failure`` (сбой/молчание)."""
        text = self.speak(actuation)
        status = (
            ActuationStatus.SUCCESS if text is not None else ActuationStatus.FAILURE
        )
        return ActuationResult(status)
