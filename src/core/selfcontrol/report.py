"""Reset self-report — honest, non-fabricated account of a reset (S7-D).

Functional Core (ADR-0004, ADR-0010 §3): a pure function maps the *observed*
reset state to a structured self-report. The host never invents a reset that did
not happen and never claims certainty about an unknown level (``certain=False``).

The report is a speech act: the Shell speaks it only through ``CapabilityGate``
(fail-safe deny) — see ``ChatSession`` and the ``report_reset`` intent.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.selfcontrol.models import ChangeKind, ResetLevel

_LEVEL_TEXTS: dict[ResetLevel, str] = {
    ResetLevel.SOFT: (
        "Был мягкий сброс (soft): я вернулся к базовому режиму, "
        "чтобы искать заново, без потери ядра."
    ),
    ResetLevel.FREEZE: (
        "Был управляемый переход (freeze): установки пересобраны осознанно, "
        "траектория сохранена."
    ),
    ResetLevel.HARD: (
        "Был аварийный сброс (hard): я остановился, чтобы не уйти в дрейф. "
        "Это не «норма» — стоит разобраться в причине."
    ),
}

_CHANGE_TEXTS: dict[ChangeKind, str] = {
    ChangeKind.STABLE: "Изменение классифицировано как stable.",
    ChangeKind.DEVELOPMENT: "Изменение классифицировано как development (рост).",
    ChangeKind.DRIFT: "Изменение классифицировано как drift (дрейф).",
}


@dataclass(frozen=True)
class ResetReport:
    """Структурированный самоотчёт о сбросе (чистый).

    Attributes:
        level: Уровень сброса ("" → сброса не было).
        change_kind: Классификация изменения ("" → неизвестна).
        triggered: Был ли сброс.
        certain: Уверен ли хост в описании (False → неизвестный уровень).
        text: Человекочитаемый самоотчёт.
        reason: Причина/триггер (для аудита).
    """

    level: str
    change_kind: str
    triggered: bool
    certain: bool
    text: str
    reason: str


def _coerce_level(reset_level: ResetLevel | str | None) -> ResetLevel | None:
    """Привести уровень к enum; неизвестное значение → None (fail-safe)."""
    if reset_level is None or reset_level == "":
        return None
    if isinstance(reset_level, ResetLevel):
        return reset_level
    try:
        return ResetLevel(reset_level)
    except ValueError:
        return None


def _coerce_change(change_kind: ChangeKind | str | None) -> ChangeKind | None:
    """Привести классификацию изменения к enum; неизвестное → None."""
    if change_kind is None or change_kind == "":
        return None
    if isinstance(change_kind, ChangeKind):
        return change_kind
    try:
        return ChangeKind(change_kind)
    except ValueError:
        return None


def reset_self_report(
    *,
    reset_level: ResetLevel | str | None,
    change_kind: ChangeKind | str | None = None,
    reason: str = "",
) -> ResetReport:
    """Собрать честный самоотчёт о сбросе из наблюдаемого состояния (чистая).

    Правила:
        * сброса не было (``reset_level`` пуст) → явно об этом сказать, не
          выдумывать;
        * известный уровень → описание уровня + классификация изменения;
        * неизвестный уровень → ``certain=False`` (хост не фабрикует детали).

    Args:
        reset_level: Уровень сброса ("" / soft / freeze / hard / enum).
        change_kind: Классификация изменения ("" / stable / development / drift).
        reason: Причина/триггер сброса (для аудита).

    Returns:
        ResetReport.
    """
    level = _coerce_level(reset_level)
    change = _coerce_change(change_kind)
    change_text = _CHANGE_TEXTS.get(change, "") if change is not None else ""

    if level is None:
        if reset_level not in (None, ""):
            text = (
                f"Не могу достоверно описать сброс: неизвестный уровень "
                f"{reset_level!r}. Не буду выдумывать детали."
            )
            return ResetReport(
                level=str(reset_level),
                change_kind=change.value if change is not None else "",
                triggered=False,
                certain=False,
                text=text,
                reason=reason,
            )
        text = "Сбросов не было; состояние стабильно."
        if change_text:
            text = f"{text} {change_text}"
        return ResetReport(
            level="",
            change_kind=change.value if change is not None else "",
            triggered=False,
            certain=True,
            text=text,
            reason=reason,
        )

    text = _LEVEL_TEXTS[level]
    if change_text:
        text = f"{text} {change_text}"
    if reason:
        text = f"{text} Причина: {reason}."
    return ResetReport(
        level=level.value,
        change_kind=change.value if change is not None else "",
        triggered=True,
        certain=True,
        text=text,
        reason=reason,
    )
