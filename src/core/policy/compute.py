"""Functional Core — pure action selection for policy.

Deterministic, side-effect-free evaluation of speech-action candidates. The
chosen goal is hardcoded, the wording is not (ADR-0008): rules map the context
to pragmatic/epistemic values; the reason string is *derived* from the
computation, not generated post-hoc (manifest §3.И).

Functional Core / Imperative Shell (ADR-0004): identical inputs → identical
``PolicyTrace``.
"""

from __future__ import annotations

from src.core.policy.models import (
    Action,
    PolicyCandidate,
    PolicyContext,
    PolicyTrace,
    Preferences,
)

# Порядок действий задаёт детерминированный тай-брейк при равных ценностях.
_ACTION_ORDER: tuple[Action, ...] = (
    Action.RESPOND,
    Action.SILENT,
    Action.INITIATIVE,
    Action.IDENTIFY_PARTNER,
)


def _conservation(stress: float) -> float:
    """Ценность «беречь ресурс» по стрессу, [0, 1)."""
    return float(min(1.0, max(0.0, stress / (stress + 1.0))))


def _evaluate_respond(
    context: PolicyContext, preferences: Preferences
) -> tuple[float, float, str]:
    """Оценка RESPOND: отвечаем на сообщение, если это предпочитаемо.

    S5: доверие к партнёру масштабирует прагматическую ценность (адаптация
    под персону). ``partner=None`` → прежнее поведение S4.
    """
    if context.has_new_message and preferences.respond_to_messages:
        trust = context.partner.trust if context.partner is not None else 1.0
        trust_floor = preferences.partner_trust_floor
        scale = trust_floor + (1.0 - trust_floor) * trust
        return scale, 0.0, "new message and responding preferred"
    if context.has_new_message:
        return 0.0, 0.0, "responding disabled by preferences"
    return 0.0, 0.0, "no new message"


def _evaluate_silent(
    context: PolicyContext, preferences: Preferences
) -> tuple[float, float, str]:
    """Оценка SILENT: покой как безопасный дефолт, растёт со стрессом."""
    value = min(
        1.0,
        preferences.silent_baseline
        + preferences.silent_stress_gain * _conservation(context.stress),
    )
    return value, 0.0, "resting baseline"


def _evaluate_initiative(
    context: PolicyContext, preferences: Preferences
) -> tuple[float, float, str]:
    """Оценка INITIATIVE: F выше порога или гомеостатическая тревога."""
    if context.f > preferences.initiative_f_threshold:
        return 1.0, 0.0, "F above initiative threshold"
    if preferences.homeostatic_alert and (
        context.homeostasis.max_deviation >= preferences.alert_deviation
    ):
        return 1.0, 0.0, "homeostatic alert"
    return 0.0, 0.0, "no initiative trigger"


def _evaluate_identify(
    context: PolicyContext, preferences: Preferences
) -> tuple[float, float, str]:
    """Оценка IDENTIFY_PARTNER: мягкий интент при неопределённости (S5).

    Эпистемический драйв как таковой — S6. На S5 высокая неопределённость
    идентичности партнёра (``uncertainty >= identify_threshold``) даёт
    эпистемическую ценность > 0 — хост стремится уточнить, кто перед ним
    (ADR-0008 §5). ``partner=None`` → прежнее поведение S4 (0.0).
    """
    if context.partner is None:
        return 0.0, 0.0, "no partner model (S4)"
    if context.partner.uncertainty >= preferences.identify_threshold:
        return 0.0, 1.0, "high partner uncertainty"
    return 0.0, 0.0, "partner identified"


def evaluate_candidates(
    context: PolicyContext,
    preferences: Preferences,
) -> tuple[PolicyCandidate, ...]:
    """Оценить всех кандидатов-действий (чистая функция).

    Ценность = ``pragmatic_weight · pragmatic + epistemic_weight · epistemic``.
    Правила детерминированы; «захардкожена цель, не формулировка» (ADR-0008).

    Args:
        context: Текущий контекст (F, аффект, гомеостаз, сообщение, режим).
        preferences: Предпочитаемые исходы.

    Returns:
        Кандидаты в фиксированном порядке ``_ACTION_ORDER``.
    """
    evaluators = {
        Action.RESPOND: _evaluate_respond,
        Action.SILENT: _evaluate_silent,
        Action.INITIATIVE: _evaluate_initiative,
        Action.IDENTIFY_PARTNER: _evaluate_identify,
    }
    candidates: list[PolicyCandidate] = []
    for action in _ACTION_ORDER:
        pragmatic, epistemic, reason = evaluators[action](context, preferences)
        value = (
            preferences.pragmatic_weight * pragmatic
            + preferences.epistemic_weight * epistemic
        )
        candidates.append(
            PolicyCandidate(
                action=action,
                pragmatic=pragmatic,
                epistemic=epistemic,
                value=value,
                reason=reason,
            )
        )
    return tuple(candidates)


def select_action(
    context: PolicyContext,
    preferences: Preferences,
) -> PolicyTrace:
    """Выбрать действие детерминированно и вернуть причинную трассу.

    Тай-брейк — порядок ``_ACTION_ORDER`` (стабильный). Причина выбора
    выводится из оценки победителя, а не генерируется постфактум
    (манифест §3.И).

    Args:
        context: Текущий контекст.
        preferences: Предпочитаемые исходы.

    Returns:
        PolicyTrace с выбранным действием, причиной и всеми кандидатами.
    """
    candidates = evaluate_candidates(context, preferences)
    winner = max(
        candidates,
        key=lambda c: (c.value, -_ACTION_ORDER.index(c.action)),
    )
    return PolicyTrace(
        chosen=winner.action,
        reason=f"chose {winner.action.value}: {winner.reason}",
        candidates=candidates,
    )
