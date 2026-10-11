"""Functional Core — open option window, total scorer, deterministic choice.

ADR-0012: the decision layer sees **all** runtime options (MCP tools plus
built-ins) and scores any option — including one it has never seen — without
code written for it. Unknown options degrade to the epistemic default
(``pragmatic = 0``, ``epistemic = uncertainty``), never crash. Enrichment
(``relevance``) is additive: with it, epistemic value is scaled by theme
binding; without it, the default stands.

Stage 2 adds **facts and conditions** ("default + enrichment"): a ``Fact`` is a
named graded regularity of the world (value in [0, 1]); a ``Guard`` is a hard
condition (gates, falsifiable) and a ``Regularity`` is a soft one (cost, a
species disposition). ``evaluate_fact`` is total: an unknown fact degrades to
``fact.default``.

Stage 3 adds **effects and irreversibility as a stance**: ``Effect`` is a
symbolic fact-delta (for planning); ``classify_reversible`` turns MCP
annotations + source trust into reversibility, conservative by default (an
untrusted source never grants autonomy).

Stage 4 adds the **Behavior Tree**: ``Node`` trees are frozen data and ``tick``
is a free function — reactive (re-evaluated from the root every tick), with
``Sequence``/``Fallback``/``Condition``/``Action`` leaves. Action outcomes enter
the pure Core through ``TickContext.action_status`` (injected by the Shell);
``tick`` never executes effectors.

Stage 5 adds the **runtime generator**: ``backward_chain`` derives a ``Node``
tree from a ``Goal`` by chaining option ``effect``/``guard`` (horizon 2–3). The
tree is never hand-written — behavior is *derived*, not programmed; an
unreachable goal or an exhausted horizon degrades to a failing ``Condition``,
never a fabricated ``Action``.

Functional Core / Imperative Shell (ADR-0004): identical inputs → identical
``OptionTrace``. The window order is the tie-break, so the whole run is a
pure function of a memoized window snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.core.actuation.models import (
    Actuation,
    ActuationKind,
    ActuationPreferences,
    Fact,
    Goal,
    Guard,
    Node,
    NodeKind,
    NodeStatus,
    Option,
    OptionCandidate,
    OptionContext,
    OptionSource,
    OptionTrace,
    OptionWindow,
    PreferredOutcome,
    Regularity,
    TickContext,
    TickMemory,
    ToolAnnotations,
)
from src.mcp.probe import Affordance


def build_options(
    tools: Sequence[Affordance] = (),
    *,
    descriptions: Mapping[str, str] | None = None,
    builtin: Sequence[Option] = (),
) -> OptionWindow:
    """Построить открытое окно опций из runtime-возможностей (чистая).

    Каждый ``Affordance`` становится ``Option(source=TOOL)`` — без кода под
    конкретный тул (открытость, инвариант SPEC). Порядок входа стабилен и
    задаёт тай-брейк выбора. ``descriptions`` — аддитивное обогащение для
    привязки темы; без него окно работоспособно (дефолт).

    Args:
        tools: Доступные MCP-тулы (runtime-вид ``tools/list``).
        descriptions: Описания тулов по имени (обогащение, этап 2).
        builtin: Встроенные опции (этап 2+; сейчас обычно пусто).

    Returns:
        OptionWindow: тул-опции в порядке входа, затем builtin.

    Raises:
        ValueError: При дубликате id в окне (валидация OptionWindow).
    """
    desc = descriptions or {}
    options = [
        Option(
            id=f"tool:{affordance.name}",
            source=OptionSource.TOOL,
            target=affordance.name,
            description=desc.get(affordance.name, ""),
            reversible=affordance.reversible,
        )
        for affordance in tools
    ]
    options.extend(builtin)
    return OptionWindow(tuple(options))


def _epistemic_default(
    context: OptionContext,
    option: Option,
    preferences: ActuationPreferences,
) -> tuple[float, str]:
    """Эпистемический дефолт: неизвестная опция ценится как зонд неопределённости.

    Без обогащения — ``uncertainty``; при ``relevance >= floor`` —
    ``uncertainty · relevance`` (привязка темы усиливает/ослабляет).
    """
    if option.relevance is not None and option.relevance >= preferences.relevance_floor:
        return (
            context.uncertainty * option.relevance,
            "epistemic option, scaled by theme relevance",
        )
    return context.uncertainty, "epistemic option (default)"


def score_option(
    option: Option,
    context: OptionContext,
    preferences: ActuationPreferences,
) -> tuple[float, float, str]:
    """Оценить опцию тотально — определена для любой опции (чистая).

    Этап 1: прагматическая ценность не объявлена ни у какого источника
    (эффекты — этап 3), поэтому ``pragmatic = 0``; эпистемическая — дефолт
    по неопределённости с аддитивным обогащением ``relevance``. У
    ``BUILTIN`` pragmatic появится на этапе 2 (стык с ``core/policy``) —
    сейчас тот же дефолт, но с иным reason (аудит).

    Этап 6 (аддитивно): если ``context.goal`` задан, прагматическая ценность —
    доля закрытия текущего разрыва до цели (``effect_gain``). Без цели
    поведение этапов 1–5 сохраняется (``pragmatic = 0``).

    Args:
        option: Любая опция (известная или нет — не падает).
        context: Состояние для оценки (неопределённость, задача, режим, цель).
        preferences: Веса/пороги скорера.

    Returns:
        ``(pragmatic, epistemic, reason)``.
    """
    epistemic, reason = _epistemic_default(context, option, preferences)
    if option.source is OptionSource.BUILTIN:
        reason = f"builtin pragmatic undefined (stage 2): {reason}"
    if context.goal is not None:
        pragmatic = effect_gain(option, context.goal, context.facts)
        if pragmatic > 0.0:
            reason = f"pragmatic toward {context.goal.fact.name}: {reason}"
        return pragmatic, epistemic, reason
    return 0.0, epistemic, reason


_DEFAULT_PREFERENCES = ActuationPreferences()


def select_option(
    window: OptionWindow,
    context: OptionContext,
    preferences: ActuationPreferences = _DEFAULT_PREFERENCES,
) -> OptionTrace:
    """Выбрать опцию детерминированно и вернуть полную трассу (чистая).

    ``value = pragmatic_weight·pragmatic + epistemic_weight·epistemic
    − cost_weight·cost − irreversible_penalty`` (для необратимых). Тай-брейк
    — индекс в окне (стабильный). В ``candidates`` — **все** опции, включая
    заниженные необратимые: видимость ≠ авторизация (инвариант SPEC).

    Args:
        window: Окно опций (снимок runtime-возможностей).
        context: Состояние для оценки.
        preferences: Веса/пороги скорера.

    Returns:
        OptionTrace: выбранная опция (None при пустом окне), причина,
        все кандидаты.
    """
    candidates: list[OptionCandidate] = []
    for option in window.options:
        pragmatic, epistemic, reason = score_option(option, context, preferences)
        penalty = 0.0 if option.reversible else preferences.irreversible_penalty
        value = (
            preferences.pragmatic_weight * pragmatic
            + preferences.epistemic_weight * epistemic
            - preferences.cost_weight * option.cost
            - penalty
        )
        candidates.append(
            OptionCandidate(
                option=option,
                pragmatic=pragmatic,
                epistemic=epistemic,
                value=value,
                reason=reason,
            )
        )
    if not candidates:
        return OptionTrace(chosen=None, reason="empty window", candidates=())
    winner = max(
        candidates,
        key=lambda c: (c.value, -window.ids.index(c.option.id)),
    )
    return OptionTrace(
        chosen=winner.option,
        reason=f"chose {winner.option.id}: {winner.reason}",
        candidates=tuple(candidates),
    )


def evaluate_fact(fact: Fact, state: Mapping[str, float]) -> float:
    """Прочитать значение факта из снимка мира, тотально (чистая).

    Неизвестный факт (нет ключа в ``state``) деградирует к ``fact.default``, а
    не падает: кондишены определены для любого факта (инвариант SPEC). Значение
    из ``state`` зажимается в [0, 1] — факт градуирован.

    Args:
        fact: Факт (несёт имя и дефолт).
        state: Снимок мира ``имя → значение`` (измеряет Shell).

    Returns:
        Значение факта в [0, 1].
    """
    value = state.get(fact.name, fact.default)
    return min(1.0, max(0.0, value))


def guard_holds(guard: Guard, state: Mapping[str, float]) -> bool:
    """Проверить жёсткий кондишен: ``fact >= threshold`` (чистая).

    Args:
        guard: Жёсткий кондишен (факт + порог).
        state: Снимок мира.

    Returns:
        True, если факт достиг порога.
    """
    return evaluate_fact(guard.fact, state) >= guard.threshold


def regularity_cost(
    regularities: Sequence[Regularity], state: Mapping[str, float]
) -> float:
    """Стоимость от мягких кондишенов: ``Σ weight · fact`` (чистая).

    Мягкое не гейтит, а удорожает действие (видовая склонность). При пустом
    списке — ``0.0`` (дефолт без обогащения). Монотонна по фактам: веса ≥ 0.

    Args:
        regularities: Мягкие кондишены.
        state: Снимок мира.

    Returns:
        Суммарная стоимость >= 0.
    """
    return sum(
        regularity.weight * evaluate_fact(regularity.fact, state)
        for regularity in regularities
    )


def unsatisfied_outcomes(
    outcomes: Sequence[PreferredOutcome], facts: Mapping[str, float]
) -> tuple[PreferredOutcome, ...]:
    """Отобрать неудовлетворённые предпочтительные исходы (чистая, этап 6).

    Исход неудовлетворён, если ``evaluate_fact(fact, facts) < target``. Порядок
    — по ``priority`` убыв., тай-брейк — порядок объявления (детерминизм).
    Пустой вход → ``()`` (нет исходов — нет цели).

    Args:
        outcomes: Предпочтительные исходы.
        facts: Снимок фактов мира.

    Returns:
        Неудовлетворённые исходы по приоритету.
    """
    unsatisfied = [
        outcome
        for outcome in outcomes
        if evaluate_fact(outcome.fact, facts) < outcome.target
    ]
    unsatisfied.sort(key=lambda outcome: -outcome.priority)
    return tuple(unsatisfied)


def select_goal(
    outcomes: Sequence[PreferredOutcome], facts: Mapping[str, float]
) -> Goal | None:
    """Выбрать цель-исход из предпочтений и снимка фактов (чистая, этап 6).

    Цель **не выводится из тега колонки** — она формируется из сочетания
    предпочтительных исходов и текущих фактов (ADR-0014). Берётся первый
    неудовлетворённый исход по приоритету; если все удовлетворены или исходов
    нет → ``None`` (действия не запускаются).

    Args:
        outcomes: Предпочтительные исходы.
        facts: Снимок фактов мира.

    Returns:
        Goal(fact, target) или None.
    """
    pending = unsatisfied_outcomes(outcomes, facts)
    if not pending:
        return None
    best = pending[0]
    return Goal(best.fact, best.target)


def goal_reached(goal: Goal, facts: Mapping[str, float]) -> bool:
    """Проверить достижение цели по снимку фактов (чистая, этап 6).

    Эквивалентно ``guard_holds(Guard(goal.fact, goal.value), facts)``:
    наблюдаемое значение факта достигло целевого.

    Args:
        goal: Цель-исход.
        facts: Снимок фактов мира.

    Returns:
        True, если факт достиг целевого значения.
    """
    return guard_holds(Guard(goal.fact, goal.value), facts)


def effect_gain(option: Option, goal: Goal, facts: Mapping[str, float]) -> float:
    """Доля закрытия текущего разрыва до цели (чистая, этап 6, ADR-0015).

    ``effect.value`` — **прогнозируемое итоговое значение факта** ∈ [0, 1]
    (не приращение), поэтому ``gap_after = goal.value − effect.value``
    применяется напрямую. Чужой эффект (``effect.fact.name != goal.fact.name``)
    или отсутствие эффекта → ``0.0``; уже достигнутая цель
    (``gap_before <= ε``) → ``0.0``.

    Args:
        option: Опция с (возможным) объявленным эффектом.
        goal: Выбранная цель-исход.
        facts: Снимок фактов мира.

    Returns:
        Доля закрытия разрыва ∈ [0, 1].
    """
    effect = option.effect
    if effect is None or effect.fact.name != goal.fact.name:
        return 0.0
    current = evaluate_fact(goal.fact, facts)
    gap_before = goal.value - current
    if gap_before <= _GAIN_EPS:
        return 0.0
    gap_after = goal.value - effect.value
    return min(1.0, max(0.0, (gap_before - gap_after) / gap_before))


_GAIN_EPS = 1e-9


def _option_value(
    option: Option, context: OptionContext, preferences: ActuationPreferences
) -> float:
    """Итоговая скалярная ценность опции — единый расчёт скорера (этап 6).

    Та же формула, что в ``select_option``: используется генератором, чтобы
    ранжировать опции по единому скаляру, а не по отдельному правилу.
    """
    pragmatic, epistemic, _ = score_option(option, context, preferences)
    penalty = 0.0 if option.reversible else preferences.irreversible_penalty
    return (
        preferences.pragmatic_weight * pragmatic
        + preferences.epistemic_weight * epistemic
        - preferences.cost_weight * option.cost
        - penalty
    )


def classify_reversible(annotations: ToolAnnotations, *, trusted: bool) -> bool:
    """Классифицировать обратимость тула: поза, не факт (чистая).

    Необратимость нельзя доказать — поэтому дефолт «необратимо», а обогащение
    может только повысить права (ADR-0012 §4). Аннотации MCP — **подсказки**:
    от недоверенного источника их нельзя принимать как основание для автономии.

    Core не знает ``Provenance`` (без цикла зависимостей): Shell транслирует
    ``official → trusted=True``, ``community/local/нет → trusted=False``.

    Args:
        annotations: MCP-аннотации тула (консервативные дефолты).
        trusted: Доверенный ли источник (градуируется Shell из provenance).

    Returns:
        True, если тул можно считать обратимым (безопасным для автономии).
    """
    if not trusted:
        return False
    return annotations.read_only_hint and not annotations.destructive_hint


def order_children(
    children: Sequence[Node], facts: Mapping[str, float]
) -> tuple[Node, ...]:
    """Упорядочить детей по мягким предпочтениям (чистая, этап 4).

    Сортировка устойчивая по ``regularity_cost`` детей (меньше — приоритетнее);
    тай-брейк — исходный порядок (детерминизм). Применяется в ``FALLBACK``;
    ``SEQUENCE`` сохраняет объявленный порядок.

    Args:
        children: Дети композита в объявленном порядке.
        facts: Снимок фактов мира (для ``regularity_cost``).

    Returns:
        Кортеж детей, отсортированный по возрастанию стоимости.
    """
    return tuple(
        sorted(children, key=lambda child: regularity_cost(child.regularities, facts))
    )


def _tick_sequence(node: Node, context: TickContext) -> tuple[NodeStatus, TickMemory]:
    """Провести SEQUENCE: все дети по порядку (чистая, этап 4).

    Args:
        node: Узел SEQUENCE.
        context: Вход tick (факты + исходы действий).

    Returns:
        Пара (статус, память): ``FAILURE`` при первом провале, ``RUNNING`` на
        первом бегущем ребёнке, иначе ``SUCCESS``.
    """
    for index, child in enumerate(node.children):
        status, child_memory = tick(child, context)
        if status is NodeStatus.FAILURE:
            return NodeStatus.FAILURE, TickMemory()
        if status is NodeStatus.RUNNING:
            return NodeStatus.RUNNING, TickMemory(
                running_path=(index,) + child_memory.running_path
            )
    return NodeStatus.SUCCESS, TickMemory()


def _tick_fallback(node: Node, context: TickContext) -> tuple[NodeStatus, TickMemory]:
    """Провести FALLBACK: первый успешный по приоритету (чистая, этап 4).

    Дети упорядочиваются по мягким предпочтениям; более приоритетный ребёнок,
    ставший ``SUCCESS``/``RUNNING``, вытесняет бегущего (преемпция).

    Args:
        node: Узел FALLBACK.
        context: Вход tick (факты + исходы действий).

    Returns:
        Пара (статус, память): ``SUCCESS``/``RUNNING`` первого преуспевшего
        ребёнка, иначе ``FAILURE``.
    """
    for index, child in enumerate(order_children(node.children, context.facts)):
        status, child_memory = tick(child, context)
        if status is NodeStatus.SUCCESS:
            return NodeStatus.SUCCESS, TickMemory()
        if status is NodeStatus.RUNNING:
            return NodeStatus.RUNNING, TickMemory(
                running_path=(index,) + child_memory.running_path
            )
    return NodeStatus.FAILURE, TickMemory()


def tick(node: Node, context: TickContext) -> tuple[NodeStatus, TickMemory]:
    """Провести один тик дерева с узла (чистая, реактивная, этап 4).

    Дерево перерешается с корня каждый тик — никакого управляющего состояния
    между тиками; решение полностью определяется ``context``. Исходы действий
    входят через ``context.action_status`` (инжектит Shell); ``tick`` не
    исполняет эффекторы.

    Args:
        node: Текущий узел (обычно корень).
        context: Вход tick (факты + исходы действий).

    Returns:
        Пара (статус узла, память с путём до бегущего листа).
    """
    if node.kind is NodeKind.CONDITION:
        guard = node.guard
        if guard is None:  # недостижимо: валидация Node
            raise ValueError("condition node requires a guard")
        status = (
            NodeStatus.SUCCESS
            if guard_holds(guard, context.facts)
            else NodeStatus.FAILURE
        )
        return status, TickMemory()
    if node.kind is NodeKind.ACTION:
        actuation = node.actuation
        if actuation is None:  # недостижимо: валидация Node
            raise ValueError("action node requires an actuation")
        status = context.action_status.get(actuation.goal, NodeStatus.RUNNING)
        return status, TickMemory()
    if node.kind is NodeKind.SEQUENCE:
        return _tick_sequence(node, context)
    return _tick_fallback(node, context)


def _goal_condition(goal: Goal) -> Node:
    """Собрать Condition-узел на цель (безопасный отказ/успех, этап 5)."""
    return Node(
        NodeKind.CONDITION, name=goal.fact.name, guard=Guard(goal.fact, goal.value)
    )


def _action_for(option: Option) -> Node:
    """Собрать Action-узел из опции (этап 5).

    ``goal`` — id опции (неймспейс окна: ``tool:<name>``); ``payload`` — реальное
    имя вызова (``option.target``, имя MCP-тула), иначе id. Эффектор адресует
    аффорданс по payload, поэтому в карте аффордансов хранится ``target``.
    """
    kind = (
        ActuationKind.INVOKE_TOOL
        if option.source is OptionSource.TOOL
        else ActuationKind.SPEAK
    )
    return Node(
        NodeKind.ACTION,
        name=option.id,
        actuation=Actuation(kind, option.id, option.target or option.id),
    )


def _find_option(
    goal: Goal,
    options: Sequence[Option],
    *,
    context: OptionContext | None = None,
    preferences: ActuationPreferences | None = None,
) -> Option | None:
    """Опция, достигающая факта цели (тай-брейк — порядок окна, этап 5/6).

    При наличии ``context``/``preferences`` подходящие опции ранжируются по
    **той же итоговой скалярной ценности**, что и ``select_option`` (единый
    расчёт, а не отдельное правило); ``context.goal`` для подцели — сама
    подцель. Без них — первая по порядку окна (прежнее поведение, тай-брейк —
    окно).
    """
    matching = [
        option
        for option in options
        if option.effect is not None and option.effect.fact.name == goal.fact.name
    ]
    if not matching:
        return None
    if context is None or preferences is None:
        return matching[0]
    subgoal_context = OptionContext(
        uncertainty=context.uncertainty,
        task=context.task,
        mode=context.mode,
        goal=goal,
        facts=context.facts,
    )
    return max(
        matching,
        key=lambda option: (
            _option_value(option, subgoal_context, preferences),
            -options.index(option),
        ),
    )


def backward_chain(
    goal: Goal,
    options: Sequence[Option],
    state: Mapping[str, float],
    *,
    max_depth: int = 3,
    context: OptionContext | None = None,
    preferences: ActuationPreferences | None = None,
) -> Node:
    """Вывести дерево от цели обратным выводом (чистая, этап 5).

    Дерево **не пишется руками** — оно выводится по ``effect``/``guard`` опций
    (ADR-0012 §6): если опция, достигающая факта цели, несёт ``guard``, его
    факт становится подцелью, которая разрешается рекурсивно и **предваряет**
    действие (``Sequence``). Уже истинная цель, недостижимость или исчерпание
    горизонта дают ``Condition``-узел на цель — честный отказ, не фиктивное
    действие.

    Этап 6 (аддитивно): при наличии ``context``/``preferences`` подходящие
    опции ранжируются по единому скаляру ``select_option``; при рекурсивном
    выборе для подцели ``effect_gain`` считается **относительно подцели**
    (``_find_option`` подставляет ``Goal(subgoal)``). Без них поведение
    прежнее (тай-брейк — порядок окна).

    Args:
        goal: Цель-исход (факт + целевое значение).
        options: Окно опций (порядок задаёт тай-брейк).
        state: Снимок фактов мира (что уже истинно).
        max_depth: Горизонт вложенности (>= 1), защита от chattering/цикла.
        context: Контекст оценки (для ранжирования); None → прежнее поведение.
        preferences: Веса скорера; None → прежнее поведение.

    Returns:
        Корень выведенного дерева.

    Raises:
        ValueError: Если ``max_depth < 1``.
    """
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")

    def derive(current: Goal, depth: int) -> Node:
        if guard_holds(Guard(current.fact, current.value), state):
            return _goal_condition(current)
        option = _find_option(
            current, options, context=context, preferences=preferences
        )
        if option is None or depth >= max_depth:
            return _goal_condition(current)
        effect = option.effect
        if effect is None:  # недостижимо: _find_option гарантирует effect
            return _goal_condition(current)
        action = _action_for(option)
        guard = option.guard
        if guard is None:
            return action
        subtree = derive(Goal(guard.fact, guard.threshold), depth + 1)
        return Node(NodeKind.SEQUENCE, name=option.id, children=(subtree, action))

    return derive(goal, 0)
