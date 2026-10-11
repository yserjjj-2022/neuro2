"""A/B stand — детерминированная тест-фикстура целевого контура (S8, проход 2).

Имена A/B живут **только здесь** (SPEC §10): в общем скорере, ``HostLoop``,
``GoalDirector`` и config специальных веток под A/B нет. Стенд честно проверяет
выбор и исполнение на **явной модели среды** (ручное объявление предпочтений и
эффектов для MVP нормально — оно не доказывает обучение).

``AbEnvironment`` — источник снимка фактов (``observe``) и транспорт
зондирования (``probe_fn``): вызов ``service_a``/``service_b`` меняет
соответствующий флаг. Три различимых исхода отказа:

- ``deny`` — оператор **отозвал право** (Gate deny): запрос не доходит до
  транспорта, ``calls`` пуст;
- ``forbid`` — Gate разрешил, но **сервис отклонил** запрос: вызов состоялся
  (в ``calls``), состояние не изменилось;
- ``lie`` — вызов «успешен», но состояние не меняется (успех вызова ≠ успех
  цели, SPEC §11).

Это три разных вида отказа/успеха, и их нельзя смешивать: «запрет до вызова» —
про авторизацию, «отказ сервиса» — про исполнение.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.core.actuation import Effect, Fact, PreferredOutcome
from src.host.gate import Capability, CapabilityGate
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


def ab_gate_denying(service: str) -> CapabilityGate:
    """Gate, отзывающий право действия у ``service`` (deny до вызова).

    Отзывает ``ACT_REVERSIBLE``: запрос на любой обратимый сервис получит
    ``GateDecision(allowed=False)`` и **не** дойдёт до транспорта (``calls``
    останется пустым).
    """
    del service  # deny моделируется отзывом права на класс действий
    gate = CapabilityGate()
    gate.revoke(Capability.ACT_REVERSIBLE)
    return gate


class AbEnvironment:
    """Среда стенда: снимок фактов + транспорт сервисов.

    Attributes:
        forbid: Имена сервисов, которые **отклонил сам сервис** (Gate разрешил).
        lie: Инструмент сообщает об успехе, но состояние не меняется (§11).
        calls: Имена реально дошедших до транспорта сервисов (для контроля).
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
        """Транспорт сервиса: меняет флаг (если не ``lie``); ``forbid`` → отказ.

        Вызывается **только** если Gate разрешил запрос (см. ``ab_gate_denying``):
        ``calls`` фиксирует факт вызова, поэтому Gate deny даёт пустой ``calls``,
        а отказ сервиса — запись в ``calls`` без изменения состояния.
        """
        self.calls.append(affordance.name)
        if affordance.name in self.forbid:
            raise RuntimeError(f"service refused: {affordance.name}")
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
