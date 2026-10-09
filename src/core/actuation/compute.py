"""Functional Core — open option window, total scorer, deterministic choice.

ADR-0012: the decision layer sees **all** runtime options (MCP tools plus
built-ins) and scores any option — including one it has never seen — without
code written for it. Unknown options degrade to the epistemic default
(``pragmatic = 0``, ``epistemic = uncertainty``), never crash. Enrichment
(``relevance``) is additive: with it, epistemic value is scaled by theme
binding; without it, the default stands.

Functional Core / Imperative Shell (ADR-0004): identical inputs → identical
``OptionTrace``. The window order is the tie-break, so the whole run is a
pure function of a memoized window snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.core.actuation.models import (
    ActuationPreferences,
    Option,
    OptionCandidate,
    OptionContext,
    OptionSource,
    OptionTrace,
    OptionWindow,
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

    Args:
        option: Любая опция (известная или нет — не падает).
        context: Состояние для оценки (неопределённость, задача, режим).
        preferences: Веса/пороги скорера.

    Returns:
        ``(pragmatic, epistemic, reason)``.
    """
    epistemic, reason = _epistemic_default(context, option, preferences)
    if option.source is OptionSource.BUILTIN:
        reason = f"builtin pragmatic undefined (stage 2): {reason}"
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
