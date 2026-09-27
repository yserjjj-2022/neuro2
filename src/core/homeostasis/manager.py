"""Imperative Shell — Homeostat evaluates interoceptive signals.

Reads raw ``SignalSource`` severities (from the sensory bus) and compares them
to configured setpoints. It owns the setpoints and the reflex threshold, but
makes no decisions: it only reports a ``HomeostasisState`` that policy and the
reflex path consume. Channels without a setpoint are ignored.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.core.homeostasis.compute import setpoint_deviation
from src.core.homeostasis.models import (
    HomeostasisState,
    HomeostaticSignal,
    Setpoint,
)
from src.mcp import SignalSource


class Homeostat:
    """Shell: оценивает интероцептивные сигналы относительно сетпоинтов.

    Attributes:
        setpoints: Сетпоинты каналов (в порядке оценки).
        reflex_threshold: Порог severity для ``is_critical``.
    """

    def __init__(
        self,
        setpoints: Sequence[Setpoint],
        reflex_threshold: float = 0.9,
    ) -> None:
        """Создать гомеостат.

        Args:
            setpoints: Сетпоинты каналов. Порядок сохраняется в результате.
            reflex_threshold: Порог severity для критического сигнала, [0, 1].

        Raises:
            ValueError: Если setpoints пуст или reflex_threshold вне [0, 1].
        """
        if not setpoints:
            raise ValueError("setpoints must not be empty")
        if not 0.0 <= reflex_threshold <= 1.0:
            raise ValueError(
                f"reflex_threshold must be in [0, 1], got {reflex_threshold}"
            )
        self.setpoints: tuple[Setpoint, ...] = tuple(setpoints)
        self.reflex_threshold = reflex_threshold

    def evaluate(self, signals: Sequence[SignalSource]) -> HomeostasisState:
        """Оценить сигналы шины относительно сетпоинтов.

        Каналы без сетпоинта игнорируются. Порядок результата — порядок
        сетпоинтов (детерминирован).

        Args:
            signals: Сигналы последнего тика (``SignalBus.last_signals``).

        Returns:
            HomeostasisState: оценки по каналам + агрегаты.
        """
        by_tag = {signal.tag: signal for signal in signals}
        evaluated: list[HomeostaticSignal] = []
        for setpoint in self.setpoints:
            signal = by_tag.get(setpoint.tag)
            severity = float(signal.severity) if signal is not None else 0.0
            evaluated.append(
                HomeostaticSignal(
                    tag=setpoint.tag,
                    severity=severity,
                    deviation=setpoint_deviation(
                        severity, setpoint.comfort, setpoint.critical
                    ),
                    is_critical=severity >= self.reflex_threshold,
                )
            )

        max_deviation = max((s.deviation for s in evaluated), default=0.0)
        severity = max((s.severity for s in evaluated), default=0.0)
        return HomeostasisState(
            signals=tuple(evaluated),
            max_deviation=max_deviation,
            severity=severity,
            is_critical=any(s.is_critical for s in evaluated),
        )
