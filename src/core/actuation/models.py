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

from collections.abc import Mapping
from dataclasses import dataclass, field
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
        guard: Жёсткий кондишен (этап 3); None → без обогащения.
        effect: Символьный эффект (этап 3); None → эпистемическая опция.
    """

    id: str
    source: OptionSource
    description: str = ""
    reversible: bool = False
    cost: float = 0.0
    relevance: float | None = None
    guard: Guard | None = None
    effect: Effect | None = None

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


# Нейтральное значение неизвестного факта (этап 2).
DEFAULT_FACT_VALUE = 0.0


@dataclass(frozen=True)
class Fact:
    """Именованная градуированная закономерность мира (этап 2).

    Значение факта ∈ [0, 1] (0/1 = булево) живёт в ``state`` — снимке мира,
    который измеряет Shell; ``Fact`` несёт лишь смысл и дефолт. Словарь фактов
    **открыт**: новый факт вводится без правки типов.

    Attributes:
        name: Стабильное имя ("network_available") — ключ в ``state``.
        default: Значение, если факт не измерен (деградация к дефолту).
        description: Смысл факта — для трассы/аудита.
    """

    name: str
    default: float = DEFAULT_FACT_VALUE
    description: str = ""

    def __post_init__(self) -> None:
        """Валидация: непустое имя, default в [0, 1].

        Raises:
            ValueError: Если имя пусто или default вне [0, 1].
        """
        if not self.name:
            raise ValueError("fact name must not be empty")
        if not 0.0 <= self.default <= 1.0:
            raise ValueError(f"fact default must be in [0, 1], got {self.default}")


@dataclass(frozen=True)
class Guard:
    """Жёсткий кондишен: закономерность мира, гейтящая действие (этап 2).

    ``fact >= threshold`` → узел проходит. Опровержим против мира (тул
    *действительно* требует сети?) — поэтому гейт, а не стоимость (ADR-0012).

    Attributes:
        fact: Проверяемый факт.
        threshold: Порог в [0, 1].
    """

    fact: Fact
    threshold: float

    def __post_init__(self) -> None:
        """Валидация: threshold в [0, 1].

        Raises:
            ValueError: Если threshold вне [0, 1].
        """
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")


@dataclass(frozen=True)
class Regularity:
    """Мягкий кондишен: видовая склонность → стоимость, не гейт (этап 2).

    Степень факта входит в стоимость действия (не запрещает). Не опровержим
    против мира — это «сложилось так», поэтому стоимость (ADR-0012).

    Attributes:
        fact: Учитываемый факт.
        weight: Вклад в стоимость, >= 0.
    """

    fact: Fact
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Валидация: weight >= 0.

        Raises:
            ValueError: Если weight < 0.
        """
        if self.weight < 0.0:
            raise ValueError(f"weight must be >= 0, got {self.weight}")


@dataclass(frozen=True)
class Effect:
    """Символьный эффект действия: ``fact := value`` (этап 3).

    Отвечает на вопрос генератора (этап 5) «достигает ли действие цели?».
    Данные-результат — это ``ActuationResult`` (Shell, этап 6); смешивать их
    с эффектом не нужно.

    Attributes:
        fact: Факт, который действие делает истинным.
        value: Целевое значение факта ∈ [0, 1].
    """

    fact: Fact
    value: float = 1.0

    def __post_init__(self) -> None:
        """Валидация: value в [0, 1].

        Raises:
            ValueError: Если value вне [0, 1].
        """
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"effect value must be in [0, 1], got {self.value}")


class ActuationKind(Enum):
    """Вид актуации: речь или вызов тула (единый контракт, разный payload)."""

    SPEAK = "speak"
    INVOKE_TOOL = "invoke_tool"


class ActuationStatus(Enum):
    """Статус актуации (идиома ROS Action Server: goal/feedback/result/preempt)."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    PREEMPTED = "preempted"


@dataclass(frozen=True)
class Actuation:
    """Единая актуация: цель + payload (текст речи ИЛИ имя тула).

    Аргументы тула — забота Shell (мемоизация, этап 6), не Core.

    Attributes:
        kind: Вид актуации.
        goal: Идентификатор цели/опции ("tool:get_weather").
        payload: Текст речи или имя тула.
    """

    kind: ActuationKind
    goal: str
    payload: str = ""

    def __post_init__(self) -> None:
        """Валидация: непустая цель.

        Raises:
            ValueError: Если goal пуст.
        """
        if not self.goal:
            raise ValueError("actuation goal must not be empty")


@dataclass(frozen=True)
class ActuationResult:
    """Результат актуации: статус + данные (→ в шину, Shell этап 6).

    Attributes:
        status: Итоговый статус.
        data: Данные-результат (пусто при Running/Failure/Preempted).
    """

    status: ActuationStatus
    data: tuple[float, ...] = ()


@dataclass(frozen=True)
class ToolAnnotations:
    """MCP-аннотации тула с консервативными дефолтами (ADR-0012 §4).

    Аннотации — только **подсказки**: недоверенный источник не даёт на их
    основании автономии (см. ``classify_reversible``). Дефолты консервативны:
    неаннотированный тул считается деструктивным и открытым миру.

    Attributes:
        read_only_hint: Тул только читает (дефолт False).
        destructive_hint: Тул может разрушать (дефолт True — консервативно).
        idempotent_hint: Повторный вызов безопасен (дефолт False).
        open_world_hint: Тул взаимодействует с внешним миром (дефолт True).
    """

    read_only_hint: bool = False
    destructive_hint: bool = True
    idempotent_hint: bool = False
    open_world_hint: bool = True


class NodeKind(Enum):
    """Вид узла Behavior Tree (минимальный словарь, ADR-0012 §5)."""

    CONDITION = "condition"  # лист: проверка guard
    ACTION = "action"  # лист: актуация (статус — из контекста)
    SEQUENCE = "sequence"  # композит: все дети по порядку
    FALLBACK = "fallback"  # композит: первый успешный (priority)


class NodeStatus(Enum):
    """Статус узла BT (идиома Behavior Tree: running/success/failure)."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class Node:
    """Узел Behavior Tree — frozen-данные, tick — свободная функция (этап 4).

    Единый тип вместо иерархии классов (data-first, как весь модуль).
    Валидация формы: лист (CONDITION/ACTION) не имеет детей и несёт
    ``guard``/``actuation``; композит (SEQUENCE/FALLBACK) имеет детей и не несёт
    ``guard``/``actuation``.

    Attributes:
        kind: Вид узла.
        name: Человекочитаемое имя (для телеметрии/трассы).
        guard: Кондишен листа CONDITION.
        actuation: Актуация листа ACTION.
        children: Дети композита (порядок задаёт приоритет).
        regularities: Мягкие предпочтения порядка детей (``order_children``).
    """

    kind: NodeKind
    name: str = ""
    guard: Guard | None = None
    actuation: Actuation | None = None
    children: tuple[Node, ...] = ()
    regularities: tuple[Regularity, ...] = ()

    def __post_init__(self) -> None:
        """Валидация формы узла (лист ↔ композит).

        Raises:
            ValueError: Если форма узла не согласована с его видом.
        """
        is_leaf = self.kind in (NodeKind.CONDITION, NodeKind.ACTION)
        if is_leaf:
            if self.children:
                raise ValueError(
                    f"leaf node {self.kind.value!r} must not have children"
                )
            if self.kind is NodeKind.CONDITION and self.guard is None:
                raise ValueError("condition node requires a guard")
            if self.kind is NodeKind.ACTION and self.actuation is None:
                raise ValueError("action node requires an actuation")
        else:
            if not self.children:
                raise ValueError(
                    f"composite node {self.kind.value!r} requires children"
                )
            if self.guard is not None or self.actuation is not None:
                raise ValueError(
                    f"composite node {self.kind.value!r} must not carry guard/actuation"
                )


@dataclass(frozen=True)
class TickContext:
    """Вход tick: факты мира + исходы действий (инжектит Shell, этап 4).

    Attributes:
        facts: Снимок фактов мира (``Mapping[str, float]``), как в этапе 2.
        action_status: Исходы активаций по ``Actuation.goal`` (от Shell).
    """

    facts: Mapping[str, float] = field(default_factory=lambda: dict[str, float]())
    action_status: Mapping[str, NodeStatus] = field(
        default_factory=lambda: dict[str, NodeStatus]()
    )


@dataclass(frozen=True)
class TickMemory:
    """Память между тиками: путь до бегущего узла (реактивный BT, этап 4).

    В дереве без ``Parallel`` активна одна цепочка, поэтому одного пути
    достаточно.

    Attributes:
        running_path: Индексы детей от корня до бегущего узла.
    """

    running_path: tuple[int, ...] = ()


@dataclass(frozen=True)
class Goal:
    """Цель-исход генератора: сделать факт истинным (этап 5).

    Decision выбирает цель-исход; ``backward_chain`` разворачивает её в дерево
    (ADR-0012 §6). Цель — это не действие, а желаемое состояние мира.

    Attributes:
        fact: Факт, который нужно сделать истинным.
        value: Целевое значение факта ∈ [0, 1].
    """

    fact: Fact
    value: float = 1.0

    def __post_init__(self) -> None:
        """Валидация: value в [0, 1].

        Raises:
            ValueError: Если value вне [0, 1].
        """
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"goal value must be in [0, 1], got {self.value}")
