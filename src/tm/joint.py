"""Joint Agency — shared goals where the host backstops, not obeys (S5, §5).

A minimal manager over shared goals (host + partner): a goal lives for a TTL
and, when the partner drifts away from it, the host reacts with a reminder
rather than executing an order (manifest §3.Г). It is a light overlay on the
existing task tag — no new attractor dynamics.
"""

from __future__ import annotations

from .models import JointGoal


class JointAgency:
    """Менеджер совместных целей (Shell, S5).

    Attributes:
        default_ttl: TTL новой общей цели, тики.
    """

    def __init__(self, *, default_ttl: int = 20) -> None:
        if default_ttl < 1:
            raise ValueError(f"default_ttl must be >= 1, got {default_ttl}")
        self.default_ttl = default_ttl
        self._goal: JointGoal | None = None

    @property
    def goal(self) -> JointGoal | None:
        """Текущая общая цель или None."""
        return self._goal

    def propose(self, task: str, *, tick: int, priority: float = 0.5) -> JointGoal:
        """Предложить общую цель (хост или партнёр).

        Args:
            task: Тег общей задачи.
            tick: Текущий тик (создание).
            priority: Приоритет цели, [0, 1].

        Returns:
            Созданная JointGoal.
        """
        self._goal = JointGoal(
            task=task,
            ttl_ticks=self.default_ttl,
            priority=priority,
            created_tick=tick,
        )
        return self._goal

    def expired(self, tick: int) -> bool:
        """Истекла ли цель по TTL (без подтверждения)."""
        if self._goal is None:
            return False
        return (tick - self._goal.created_tick) >= self._goal.ttl_ticks

    def update(self, active_task: str, *, tick: int) -> str | None:
        """Проверить удержание общей цели и, при уходе, вернуть реакцию.

        Хост подстраховывает: если партнёр/задача ушли от общей цели, это
        не исполнение приказа, а напоминание. Истёкшая цель снимается.

        Args:
            active_task: Текущий тег активной задачи (аттрактор).
            tick: Текущий тик.

        Returns:
            Текст-напоминание при уходе от цели, иначе None.
        """
        if self._goal is None:
            return None
        if self.expired(tick):
            self._goal = None
            return None
        if active_task and active_task != "none" and active_task != self._goal.task:
            return (
                f"Мы договаривались о задаче «{self._goal.task}», "
                f"а сейчас «{active_task}». Вернёмся?"
            )
        return None

    def complete(self) -> None:
        """Снять общую цель (достигнута/отменена)."""
        self._goal = None
