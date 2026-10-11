"""A/B stand — детерминированная тест-фикстура целевого контура (S8, проход 2).

Имена A/B живут **только здесь** (SPEC §10): в общем скорере, ``HostLoop``,
``GoalDirector`` и config специальных веток под A/B нет. Стенд честно проверяет
выбор и исполнение на **явной модели среды** (ручное объявление предпочтений и
эффектов для MVP нормально — оно не доказывает обучение).

``AbEnvironment`` — источник снимка фактов (``observe``) и транспорт
зондирования (``probe_fn``): вызов ``service_a``/``service_b`` меняет
соответствующий флаг. Флаги ``forbid`` (оператор запретил — отказ гейта) и
``lie`` (успех без изменения состояния) моделируют два разных вида успеха
(SPEC §11).
"""

from __future__ import annotations

from collections.abc import Mapping

from src.core.actuation import Effect, Fact, PreferredOutcome
from src.mcp.models import SignalCategory
from src.mcp.probe import Affordance, AffordanceMap

A_SERVICED = Fact("a_serviced", description="Сервис A выполнен.")
B_SERVICED = Fact("b_serviced", description="Сервис B выполнен.")


def ab_affordances() -> AffordanceMap:
    """Карта аффордансов стенда: два обратимых сервиса."""
    return AffordanceMap(
        (
            Affordance("service_a", SignalCategory.EXTEROCEPTIVE, True, 1),
            Affordance("service_b", SignalCategory.EXTEROCEPTIVE, True, 1),
        )
    )


def ab_outcomes(
    *, a_priority: float = 1.0, b_priority: float = 1.0
) -> tuple[PreferredOutcome, ...]:
    """Предпочтительные исходы стенда: оба сервиса обслужены."""
    return (
        PreferredOutcome(A_SERVICED, target=1.0, priority=a_priority),
        PreferredOutcome(B_SERVICED, target=1.0, priority=b_priority),
    )


class AbEnvironment:
    """Среда стенда: снимок фактов + транспорт сервисов.

    Attributes:
        forbid: Имена сервисов, которые оператор запретил (отказ гейта).
        lie: Инструмент сообщает об успехе, но состояние не меняется (§11).
        calls: Имена реально вызванных сервисов (для контроля).
    """

    def __init__(
        self,
        *,
        a_serviced: float = 0.0,
        b_serviced: float = 0.0,
        forbid: tuple[str, ...] = (),
        lie: bool = False,
    ) -> None:
        self._facts: dict[str, float] = {
            "a_serviced": a_serviced,
            "b_serviced": b_serviced,
        }
        self.forbid = set(forbid)
        self.lie = lie
        self.calls: list[str] = []

    def observe(self) -> Mapping[str, float]:
        """Снимок фактов мира (значения ∈ [0, 1])."""
        return dict(self._facts)

    def set_fact(self, name: str, value: float) -> None:
        """Внешне изменить факт (имитация изменения среды оператором)."""
        self._facts[name] = value

    def probe_fn(self, affordance: Affordance) -> tuple[float, ...]:
        """Транспорт сервиса: меняет флаг (если не ``lie``); ``forbid`` → отказ."""
        self.calls.append(affordance.name)
        if affordance.name in self.forbid:
            raise RuntimeError(f"forbidden: {affordance.name}")
        if not self.lie:
            if affordance.name == "service_a":
                self._facts["a_serviced"] = 1.0
            elif affordance.name == "service_b":
                self._facts["b_serviced"] = 1.0
        return (1.0,)


def ab_declarations() -> tuple[tuple[str, Effect], ...]:
    """Декларации семантики опций стенда: ``tool:service_x`` → ``x_serviced := 1``."""
    return (
        ("tool:service_a", Effect(A_SERVICED, value=1.0)),
        ("tool:service_b", Effect(B_SERVICED, value=1.0)),
    )
