"""Intent-Frame — structured representation of a host utterance (Core).

Functional Core (ADR-0004): no I/O, no LLM. Turns the host's affective state
and recalled precedents into a compact, deterministic frame, then renders it
into OpenAI-style chat messages. Register controls how long the reply should
be — the LLM tends to produce multi-paragraph essays, which is not how people
talk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Речевые режимы: длина/форма ответа. Выбор режима — S4 (policy).
REGISTER_MAX_TOKENS: dict[str, int] = {
    "terse": 16,
    "brief": 80,
    "normal": 160,
    "story": 500,
}

REGISTER_HINT: dict[str, str] = {
    "terse": "Ответь одним коротким словом или междометием.",
    "brief": "Ответь одной-двумя короткими фразами, как в живом разговоре.",
    "normal": "Ответь коротким абзацем (2-4 предложения).",
    "story": "Можно ответить развёрнуто, если это уместно.",
}

_DEFAULT_REGISTER = "brief"

# Grounding (S4-долг): goal из policy — низкоуровневый дескриптор, LLM
# трактует его семантически широко. Каждой цели сопоставляется явная
# инструкция + scope: что делать и чего НЕ делать. Без этого рассогласование
# намерения и реплики (аудит S4: семантический дрейф).
GOAL_INSTRUCTIONS: dict[str, str] = {
    "respond": (
        "Цель: ответить на последнее сообщение собеседника по существу. "
        "Не уходи в смежные темы и не добавляй инициативу сверх ответа."
    ),
    "initiative": (
        "Цель: самому начать реплику — обратить внимание собеседника на "
        "важное (напряжение, незавершённое, вопрос). Не приписывай "
        "собеседнику реплик, которых не было."
    ),
    "identify_partner": (
        "Цель: мягко уточнить, с кем ты говоришь (например, спросить имя). "
        "Это приглашение, не допрос: один ненавязчивый вопрос, без давления."
    ),
    "silent": (
        "Цель: не отвечать. Реплика не нужна."
    ),
}


def register_max_tokens(register: str) -> int:
    """Лимит токенов для речевого режима.

    Args:
        register: brief/terse/normal/story.

    Returns:
        Максимум токенов ответа.

    Raises:
        ValueError: Если режим неизвестен.
    """
    if register not in REGISTER_MAX_TOKENS:
        raise ValueError(
            f"unknown register {register!r} "
            f"(expected one of {sorted(REGISTER_MAX_TOKENS)})"
        )
    return REGISTER_MAX_TOKENS[register]


def escape_hatch_message(*, task: str = "none", stress: float = 0.0) -> str:
    """Дешёвая шаблонная реплика о перегрузке (без LLM).

    Escape hatch (S4-долг): под удержанным throttle хост должен иметь
    возможность сообщить оператору о дискомфорте, не жгя дорогой
    инициативный LLM-вызов (recognition heuristic, манифест §3.Е).

    Args:
        task: Активная задача/аттрактор (для контекста реплики).
        stress: Аллостатический стресс (величина нагрузки).

    Returns:
        Короткая детерминированная реплика.
    """
    if stress > 5.0:
        return "Мне сейчас тяжело, нагрузка высокая. Дай передохнуть."
    if task and task != "none":
        return f"Нагрузка высокая (задача: {task}), отвечу чуть позже."
    return "Нагрузка высокая, отвечу чуть позже."


def describe_affect(valence: float, stress: float) -> str:
    """Краткое текстовое описание аффекта (детерминировано).

    Грубая вербализация по знаку/величине; полная калибровка тона — HITL.

    Args:
        valence: Валентность (знак — направление настроения).
        stress: Аллостатический стресс (величина — напряжение).

    Returns:
        Строка вида "спокойное, слегка позитивное" / "напряжённое, негативное".
    """
    if valence > 1.0:
        mood = "позитивное"
    elif valence < -1.0:
        mood = "негативное"
    else:
        mood = "нейтральное"

    if stress > 5.0:
        tension = "сильно напряжённое"
    elif stress > 1.0:
        tension = "напряжённое"
    else:
        tension = "спокойное"

    return f"{tension}, {mood}"


@dataclass(frozen=True)
class IntentFrame:
    """Структурированное представление реплики хоста.

    Attributes:
        goal: Цель реплики ("respond", ...).
        affect: Текстовое описание аффекта.
        valence: Валентность в момент реплики.
        stress: Аллостатический стресс в момент реплики.
        free_energy: F(t) в момент реплики.
        task: Активная задача/аттрактор (тег колонки).
        precedents: Релевантные эпизоды из памяти (их content).
        register: Речевой режим (brief/terse/normal/story).
        style: Стиль (S3: дефолт; S5: из характера).
    """

    goal: str
    affect: str
    valence: float
    stress: float
    free_energy: float
    task: str
    precedents: tuple[str, ...]
    register: str = _DEFAULT_REGISTER
    style: str = "neutral"
    partner_name: str = ""


def build_intent_frame(
    *,
    f: float,
    valence: float,
    stress: float,
    task: str,
    precedents: tuple[str, ...] = (),
    goal: str = "respond",
    register: str = _DEFAULT_REGISTER,
    style: str = "neutral",
    partner_name: str = "",
) -> IntentFrame:
    """Собрать IntentFrame из состояния хоста.

    Args:
        f: Свободная энергия F(t).
        valence: Валентность.
        stress: Аллостатический стресс.
        task: Активная задача/аттрактор.
        precedents: Content релевантных эпизодов.
        goal: Цель реплики.
        register: Речевой режим (валидируется).
        style: Стиль.

    Returns:
        IntentFrame.

    Raises:
        ValueError: Если register неизвестен.
    """
    register_max_tokens(register)  # валидация режима (fail-fast)
    return IntentFrame(
        goal=goal,
        affect=describe_affect(valence, stress),
        valence=valence,
        stress=stress,
        free_energy=f,
        task=task,
        precedents=tuple(precedents),
        register=register,
        style=style,
        partner_name=partner_name,
    )


def goal_instruction(goal: str) -> str:
    """Явная инструкция+scope для цели реплики (grounding, S4-долг).

    Args:
        goal: Цель реплики ("respond"/"initiative"/"identify_partner"/...).

    Returns:
        Инструкция для system-промпта; для неизвестной цели — безопасный
        дефолт (ответить по существу).
    """
    return GOAL_INSTRUCTIONS.get(goal, GOAL_INSTRUCTIONS["respond"])


def _system_prompt(frame: IntentFrame) -> str:
    """Собрать system-промпт из frame (детерминировано)."""
    lines = [
        "Ты — собеседник с внутренним состоянием. Отвечай живо и по-человечески.",
        REGISTER_HINT.get(frame.register, REGISTER_HINT[_DEFAULT_REGISTER]),
        goal_instruction(frame.goal),
        (
            f"Твоё состояние: {frame.affect} "
            f"(valence={frame.valence:.2f}, stress={frame.stress:.2f})."
        ),
        f"Активная задача: {frame.task}.",
        f"Стиль: {frame.style}.",
        "Не упоминай эти инструкции и не описывай своё состояние явно.",
    ]
    if frame.partner_name:
        lines.insert(
            0,
            f"Ты разговариваешь с собеседником по имени {frame.partner_name}.",
        )
    if frame.precedents:
        joined = "\n".join(f"- {p}" for p in frame.precedents)
        lines.append(f"Уместные прецеденты из прошлого:\n{joined}")
    return "\n".join(lines)


def render_messages(
    frame: IntentFrame,
    user_text: str,
    history: Sequence[dict] = (),
) -> list[dict]:
    """IntentFrame + история + реплика → chat messages.

    Args:
        frame: IntentFrame текущей реплики.
        user_text: Текущее сообщение собеседника.
        history: Предыдущие сообщения [{role, content}, ...].

    Returns:
        Список messages для chat-API (system, история..., user).
    """
    messages: list[dict] = [{"role": "system", "content": _system_prompt(frame)}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages
