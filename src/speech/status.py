"""Status line for the chat stand (Core, pure).

Renders the host's affective/attentional state as a single compact line so the
operator can validate that the reply follows the state (HITL, S3 gates). No
I/O, no formatting side effects — just a string.
"""

from __future__ import annotations


def format_status(
    *,
    f: float,
    valence: float,
    stress: float,
    gamma: float,
    task: str,
    recall_hit: bool,
    drift: bool,
    channel_contrib: tuple[tuple[str, float], ...] = (),
) -> str:
    """Собрать строку состояния хоста для диалогового стенда.

    Args:
        f: Свободная энергия F(t).
        valence: Валентность.
        stress: Аллостатический стресс.
        gamma: Агрегат precision γ.
        task: Активная задача/аттрактор (тег колонки).
        recall_hit: Найден ли релевантный прецедент в памяти.
        drift: Сработал ли детектор дрейфа.
        channel_contrib: Вклад каналов шины в F ``(tag, вклад)`` — какие
            каналы реально давят (BACKLOG). Пусто → не выводится.

    Returns:
        Строка вида ``[F=2.31 val=-1.40 stress=0.80 γ=4.20 задача=tone
        recall=1 дрейф=нет | вклад: battery=0.42 message=0.08]``.
    """
    line = (
        f"[F={f:.2f} val={valence:+.2f} stress={stress:.2f} γ={gamma:.2f} "
        f"задача={task} recall={1 if recall_hit else 0} "
        f"дрейф={'да' if drift else 'нет'}"
    )
    if channel_contrib:
        parts = " ".join(f"{tag}={value:.2f}" for tag, value in channel_contrib)
        line += f" | вклад: {parts}"
    return line + "]"
