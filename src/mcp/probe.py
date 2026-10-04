"""MCP probing — epistemic actions as gated affordances (S6 проход 2).

Manifest §3.Ж: MCP Tools — active epistemic actions, invoked when uncertainty
is cheaper to reduce by an external query than by guessing. The MCP *transport*
(resources/tools over the wire) is not implemented; this module defines the
**contract** of a probing affordance and a pure selection rule. Execution goes
through the capability gate in the Shell (``src/host/probe.py``).

Functional Core / Imperative Shell (ADR-0004): all functions here are pure.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.mcp.models import SignalCategory


@dataclass(frozen=True)
class Affordance:
    """Одно доступное эпистемическое действие (элемент карты аффордансов).

    Attributes:
        name: Имя действия ("web_search", "check_time", ...).
        category: Категория сигнала, который оно добывает.
        reversible: Обратимо ли действие (безопасно для автономии).
        dim: Размерность возвращаемого вектора данных (> 0).
    """

    name: str
    category: SignalCategory
    reversible: bool = True
    dim: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("affordance name must not be empty")
        if self.dim <= 0:
            raise ValueError(f"affordance dim must be > 0, got {self.dim}")


@dataclass(frozen=True)
class AffordanceMap:
    """Карта аффордансов — пространство того, что хост в принципе умеет.

    Attributes:
        affordances: Доступные действия (уникальные имена).
    """

    affordances: tuple[Affordance, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for affordance in self.affordances:
            if affordance.name in seen:
                raise ValueError(f"duplicate affordance: {affordance.name!r}")
            seen.add(affordance.name)

    def find(self, name: str) -> Affordance | None:
        """Найти аффорданс по имени (None, если нет)."""
        for affordance in self.affordances:
            if affordance.name == name:
                return affordance
        return None

    @property
    def names(self) -> tuple[str, ...]:
        """Имена всех аффордансов в порядке объявления."""
        return tuple(a.name for a in self.affordances)

    @property
    def reversible(self) -> tuple[Affordance, ...]:
        """Только обратимые аффордансы (кандидаты автономного зондирования)."""
        return tuple(a for a in self.affordances if a.reversible)


@dataclass(frozen=True)
class ProbeRequest:
    """Запрос на эпистемическое зондирование.

    Attributes:
        affordance: Имя аффорданса.
        reason: Причина (для аудита/трассировки).
    """

    affordance: str
    reason: str


@dataclass(frozen=True)
class ProbeResult:
    """Результат зондирования (в том числе отказ gate).

    Attributes:
        affordance: Имя аффорданса.
        success: Успешно ли выполнено (False при отказе/сбое).
        data: Возвращённый вектор данных (пусто при отказе).
        reason: Причина (успех/отказ/сбой) — для телеметрии.
    """

    affordance: str
    success: bool
    data: tuple[float, ...]
    reason: str


def default_affordances() -> AffordanceMap:
    """Стандартная карта аффордансов (mock-набор до реального MCP-транспорта).

    Returns:
        AffordanceMap с обратимыми действиями.
    """
    return AffordanceMap(
        (
            Affordance("web_search", SignalCategory.EXTEROCEPTIVE, True, 4),
            Affordance("check_time", SignalCategory.EXTEROCEPTIVE, True, 2),
            Affordance("read_sensor", SignalCategory.INTEROCEPTIVE, True, 1),
        )
    )


def select_affordance(
    uncertainty: float,
    affordances: AffordanceMap,
    *,
    threshold: float,
) -> Affordance | None:
    """Выбрать аффорданс для зондирования по неопределённости (чистая).

    Драйв мягкий: ниже порога — не зондируем. Выбор детерминирован — первый
    обратимый аффорданс в порядке карты (тай-брейк стабилен).

    Args:
        uncertainty: Метакогнитивная неопределённость, [0, 1].
        affordances: Карта аффордансов.
        threshold: Порог неопределённости для зондирования, [0, 1].

    Returns:
        Affordance или None (ниже порога / нет обратимых).

    Raises:
        ValueError: Если threshold вне [0, 1].
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if uncertainty < threshold:
        return None
    reversible = affordances.reversible
    return reversible[0] if reversible else None
