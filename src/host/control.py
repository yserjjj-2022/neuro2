"""Control channel — hot-testing command interface over HostLoop (S4).

ADR-0005 §8: operational control and HITL. The S4 minimal set is
``status``/``pause``/``resume``/``step`` — enough to observe the reflex path
tick by tick and to run the goal-directed test manually. Extension
(inject/set/snapshot/restore/freeze/kill) is deferred; ``kill`` is covered by
SIGINT.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.host.loop import HostLoop
from src.speech.status import format_status


@dataclass
class ControlChannel:
    """Локальный командный интерфейс над ``HostLoop`` (Shell).

    Владеет флагом паузы и счётчиком тиков. ``step`` продвигает loop ровно
    на n тиков, ``pause``/``resume`` управляют флагом, ``status`` рендерит
    строку состояния (переиспользует ``format_status``).

    Attributes:
        loop: Управляемый host loop.
        paused: Стоят ли тики (freeze: состояние и канал живы).
        tick: Счётчик выполненных тиков (для синтетического времени).
    """

    loop: HostLoop
    paused: bool = False
    tick: int = 0
    _steps: int = field(default=0, init=False, repr=False)

    def status(self) -> str:
        """Строка состояния хоста (для HITL-наблюдения).

        Returns:
            Строка вида ``[F=.. val=.. stress=.. γ=.. задача=.. recall=..
            дрейф=..]``.
        """
        outcome = self.loop.last_outcome
        f = outcome.result.f if outcome is not None else 0.0
        valence = outcome.result.valence if outcome is not None else 0.0
        stress = outcome.result.allostatic_stress if outcome is not None else 0.0
        gamma = outcome.result.gamma if outcome is not None else 0.0
        return format_status(
            f=f,
            valence=valence,
            stress=stress,
            gamma=gamma,
            task=self.loop.active_task(),
            recall_hit=self.loop.last_memory_hit,
            drift=self.loop.last_drift,
        )

    def pause(self) -> None:
        """Остановить тики (freeze): состояние и канал управления живы."""
        self.paused = True

    def resume(self) -> None:
        """Возобновить тики после паузы."""
        self.paused = False

    def step(self, n: int = 1) -> int:
        """Сделать ровно n тиков (пошаговое наблюдение рефлекса).

        Работает и на паузе: ``step`` — это разовый ручной тик.

        Args:
            n: Число тиков, n >= 0.

        Returns:
            Фактически выполненные тики.

        Raises:
            ValueError: Если n < 0.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        for _ in range(n):
            self.loop.step_once(self.tick)
            self.tick += 1
        self._steps += n
        return n

    def run(self, max_ticks: int = 0) -> int:
        """Гнать тики, пока не пауза или не исчерпан лимит.

        Args:
            max_ticks: Максимум тиков (0 → до паузы).

        Returns:
            Фактически выполненные тики.
        """
        executed = 0
        while not self.paused and (max_ticks == 0 or executed < max_ticks):
            self.loop.step_once(self.tick)
            self.tick += 1
            executed += 1
        return executed

    @property
    def steps(self) -> int:
        """Сколько тиков сделано через канал (аудит/наблюдаемость)."""
        return self._steps
