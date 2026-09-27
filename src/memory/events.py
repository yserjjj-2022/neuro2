"""Significance detection and event content — pure core.

Decides *when* an episode is worth remembering and builds a text descriptor
for situations that have no companion text (e.g. a free-energy spike or a
reflex interoceptive alarm).
"""

from __future__ import annotations


def is_significant_event(
    f: float,
    prev_f: float,
    reflex_tags: tuple[str, ...],
    spike_threshold: float,
    has_new_message: bool = False,
) -> bool:
    """Значимо ли событие для эпизодической памяти.

    Событие значимо, если есть критический сигнал (``reflex_tags``), ИЛИ
    F(t) подскочила относительно предыдущего тика выше порога, ИЛИ пришло
    новое сообщение оператора (коммуникативный вход событиен по природе —
    ADR-0006).

    Args:
        f: Текущее значение свободной энергии F(t).
        prev_f: Значение F(t-1).
        reflex_tags: Теги критических сигналов текущего тика.
        spike_threshold: Порог всплеска F (> 0 обычно; 0 → любой рост).
        has_new_message: Пришло ли новое сообщение оператора.

    Returns:
        True, если событие следует запомнить.

    Raises:
        ValueError: Если spike_threshold < 0.
    """
    if spike_threshold < 0.0:
        raise ValueError(f"spike_threshold must be >= 0, got {spike_threshold}")
    if reflex_tags or has_new_message:
        return True
    return (f - prev_f) > spike_threshold


def build_event_content(
    active_tags: tuple[str, ...],
    reflex_tags: tuple[str, ...],
    valence: float,
    stress: float,
) -> str:
    """Текстовый дескриптор ситуации (когда нет текста собеседника).

    Детерминированный, человекочитаемый: теги каналов + аффект. Служит
    ``content`` эпизода для событий без коммуникативного входа.

    Args:
        active_tags: Теги активных сегментов шины.
        reflex_tags: Теги критических сигналов.
        valence: Валентность в момент события.
        stress: Аллостатический стресс в момент события.

    Returns:
        Непустая строка-описание.
    """
    active = ",".join(active_tags) if active_tags else "none"
    reflex = ",".join(reflex_tags) if reflex_tags else "none"
    return (
        f"event active=[{active}] reflex=[{reflex}] "
        f"valence={valence:.3f} stress={stress:.3f}"
    )
