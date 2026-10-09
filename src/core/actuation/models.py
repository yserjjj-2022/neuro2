"""Domain objects for the actuation module (S8 stage 1).

The **option window** is an open, runtime-generated set of everything the
decision layer could do right now (ADR-0012): built-ins plus MCP tools.
Options are *not* a closed enum — a new tool becomes visible with no code
written for it. Visibility is not authorization: an irreversible option is
present and down-weighted here, but execution is gated elsewhere
(``CapabilityGate``).

Frozen dataclasses, analogous to ``policy.PolicyTrace``: immutable snapshots.
``ActuationPreferences`` carries the scoring weights; enrichment fields
(``description``, ``relevance``) are additive — defaults keep the scorer
total without them (ADR-0012 §3 "default + enrichment").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OptionSource(Enum):
    """Источник опции в окне (открытый, не закрытый класс действий)."""

    TOOL = "tool"  # MCP-тул (эпистемическое зондирование)
    BUILTIN = "builtin"  # встроенное действие (этап 2+, pragmatic не объявлен)


@dataclass(frozen=True)
class Option:
    """Одна возможность в окне выбора.

    Attributes:
        id: Стабильный идентификатор ("tool:get_weather").
        source: Источник опции (тул/встроенная).
        description: Описание — для привязки темы (этап 2) и трассы.
        reversible: Обратимость; консервативный дефолт False (ADR-0012 §4).
        cost: Стоимость/латентность (оценка, не замер), >= 0.
        relevance: Привязка темы [0, 1]; None → дефолт (без обогащения).
    """

    id: str
    source: OptionSource
    description: str = ""
    reversible: bool = False
    cost: float = 0.0
    relevance: float | None = None

    def __post_init__(self) -> None:
        """Валидация: непустой id, cost >= 0, relevance в [0, 1] или None.

        Raises:
            ValueError: Если id пуст, cost < 0 или relevance вне [0, 1].
        """
        if not self.id:
            raise ValueError("option id must not be empty")
        if self.cost < 0.0:
            raise ValueError(f"cost must be >= 0, got {self.cost}")
        if self.relevance is not None and not 0.0 <= self.relevance <= 1.0:
            raise ValueError(f"relevance must be in [0, 1], got {self.relevance}")


@dataclass(frozen=True)
class OptionWindow:
    """Открытое окно опций — снимок runtime-возможностей.

    Порядок стабилен (детерминизм: тай-брейк argmax — индекс в окне).

    Attributes:
        options: Опции в порядке построения (уникальные id).
    """

    options: tuple[Option, ...] = ()

    def __post_init__(self) -> None:
        """Валидация: уникальность id в окне.

        Raises:
            ValueError: При дубликате id.
        """
        seen: set[str] = set()
        for option in self.options:
            if option.id in seen:
                raise ValueError(f"duplicate option id: {option.id!r}")
            seen.add(option.id)

    def find(self, option_id: str) -> Option | None:
        """Найти опцию по id (None, если нет)."""
        for option in self.options:
            if option.id == option_id:
                return option
        return None

    @property
    def ids(self) -> tuple[str, ...]:
        """Id всех опций в стабильном порядке."""
        return tuple(o.id for o in self.options)

    @property
    def tools(self) -> tuple[Option, ...]:
        """Только туловые опции (MCP)."""
        return tuple(o for o in self.options if o.source is OptionSource.TOOL)


@dataclass(frozen=True)
class OptionContext:
    """Вход оценки: состояние, относительно которого оценивается опция.

    Attributes:
        uncertainty: Метакогнитивная неопределённость, [0, 1].
        task: Активная задача/аттрактор (тег колонки).
        mode: Режим хоста (game/cooperative/free).
    """

    uncertainty: float
    task: str = "none"
    mode: str = "free"

    def __post_init__(self) -> None:
        """Валидация: uncertainty в [0, 1].

        Raises:
            ValueError: Если uncertainty вне [0, 1].
        """
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError(f"uncertainty must be in [0, 1], got {self.uncertainty}")


@dataclass(frozen=True)
class ActuationPreferences:
    """Веса скорера окна (калибруются по телеметрии, ADR-0012 §10).

    Attributes:
        pragmatic_weight: Вес прагматической ценности, >= 0.
        epistemic_weight: Вес эпистемической ценности, >= 0.
        cost_weight: Штраф за стоимость, >= 0.
        irreversible_penalty: Занижение необратимой опции без consent, >= 0.
        relevance_floor: Порог привязки темы [0, 1]; ниже — не учитывается.
    """

    pragmatic_weight: float = 1.0
    epistemic_weight: float = 0.5
    cost_weight: float = 0.1
    irreversible_penalty: float = 0.5
    relevance_floor: float = 0.0

    def __post_init__(self) -> None:
        """Валидация: неотрицательные веса, relevance_floor в [0, 1].

        Raises:
            ValueError: Если любой вес < 0 или relevance_floor вне [0, 1].
        """
        for name in (
            "pragmatic_weight",
            "epistemic_weight",
            "cost_weight",
            "irreversible_penalty",
        ):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if not 0.0 <= self.relevance_floor <= 1.0:
            raise ValueError(
                f"relevance_floor must be in [0, 1], got {self.relevance_floor}"
            )


@dataclass(frozen=True)
class OptionCandidate:
    """Оценка одной опции (входит в трассу — все, включая отклонённые).

    Attributes:
        option: Оценканная опция.
        pragmatic: Прагматическая ценность (этап 1: 0 — эффект не объявлен).
        epistemic: Эпистемическая ценность (снижение неопределённости).
        value: Итоговая ценность (взвешенная сумма со штрафами).
        reason: Причина оценки (для трассировки).
    """

    option: Option
    pragmatic: float
    epistemic: float
    value: float
    reason: str


@dataclass(frozen=True)
class OptionTrace:
    """Причинная трассировка выбора в окне (explainability — инвариант).

    Attributes:
        chosen: Выбранная опция (None → окно пусто).
        reason: Причина выбора, выведенная из расчёта (не постфактум-текст).
        candidates: Все рассмотренные опции с оценками (видимость ≠ авторизация).
    """

    chosen: Option | None
    reason: str
    candidates: tuple[OptionCandidate, ...]
