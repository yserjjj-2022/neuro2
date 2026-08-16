"""Imperative Shell — VotingManager owns k and caches the last result.

Functional Core / Imperative Shell (ADR-0004):
- Core (kwta) — pure function, tested without Shell
- Shell (VotingManager) — owns k (tuning), caches the result
- In prod: vote(scores) is called from wiring after cmc.step(u)
"""

from __future__ import annotations

from .kwta import kwta
from .models import Vector, VotingResult


class VotingManager:
    """Imperative Shell: параметр k + кэш последнего результата.

    k задаётся вызывающим кодом (не зашит в модуль) — инвариант SPEC.
    Дефолт 1 = hard-WTA; распределённое представление требует k > 1.
    """

    def __init__(self, k: int = 1) -> None:
        """Инициализация.

        Args:
            k: Число победителей, 1 ≤ k. Валидируется на ≥ 1;
                верхняя граница (k ≤ N) зависит от входного scores.

        Raises:
            ValueError: Если k < 1.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k
        self._last: VotingResult | None = None

    def vote(self, scores: Vector) -> VotingResult:
        """Проголосовать: выбрать k победителей из scores.

        Args:
            scores: Вектор оценок колонок, shape (N,).

        Returns:
            VotingResult — индексы/маска/оценки победителей.

        Raises:
            ValueError: Если k > N (недостаточно колонок для k победителей)
                или scores пустой/не одномерный (делегируется в kwta).
        """
        result = kwta(scores, self._k)
        self._last = result
        return result

    @property
    def last(self) -> VotingResult | None:
        """Последний результат vote(), None до первого вызова."""
        return self._last

    def set_k(self, k: int) -> None:
        """Изменить число победителей (tuning).

        Args:
            k: Новое значение, k ≥ 1.

        Raises:
            ValueError: Если k < 1.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k = k
