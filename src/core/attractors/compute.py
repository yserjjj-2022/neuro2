"""Functional Core — pure functions for task attractor dynamics.

Contains the three decision functions that determine whether to switch
attractors, how long to hold, and whether the current state is within
the basin of attraction.

Functional Core / Imperative Shell (ADR-0004):
- All functions are pure: identical inputs → identical outputs,
  no mutation of input arrays.
"""

from __future__ import annotations


def compute_dwell(
    base_dwell: int,
    dwell_slope: float,
    plasticity_gain: float,
    score_current: float,
    score_runner_up: float,
    history_size: int,
) -> int:
    """Вычислить минимальное время удержания (гистерезис + STP).

    Формула:
        min_dwell = max(base, base + slope*Δ + gain*history)

    Семантика:
        - base_dwell — жёсткий пол удержания (не проседает). Гарантирует,
          что dwell-time механизм защищает от дребезга, а не маскирует его.
        - dwell_slope > 0: выигрыш продлевает удержание (гистерезис).
        - plasticity_gain > 0: чем дольше держусь, тем сильнее (STP).
        - Проигрыш НЕ снижает dwell ниже base — немедленное переключение
          управляется отдельным механизмом (see check_immediate_switch).

    Args:
        base_dwell: α — жёсткий пол удержания (не проседает).
        dwell_slope: β — прирост dwell при выигрыше (Δ > 0).
        plasticity_gain: γ — прирост устойчивости за каждый тик удержания (STP).
        score_current: Оценка текущей задачи (аттрактора).
        score_runner_up: Оценка претендента (топ-1 из scores, кроме held).
        history_size: Сколько тиков подряд аттрактор не менялся.

    Returns:
        min_dwell в тиках (>= base_dwell, округлено до ближайшего целого).

    Examples:
        >>> compute_dwell(5, 2.0, 0.0, 0.5, 0.3, 0)
        5
        >>> compute_dwell(5, 2.0, 0.0, 0.5, 0.8, 0)
        5
        >>> compute_dwell(5, 0.0, 0.1, 0.5, 0.5, 5)
        6
    """
    delta = score_current - score_runner_up
    raw_dwell = base_dwell + dwell_slope * delta + plasticity_gain * history_size
    return max(base_dwell, round(raw_dwell))


def check_basin_stability(
    basin_threshold: float,
    score_current: float,
    score_runner_up: float,
    history_size: int,
) -> bool:
    """Проверить, находится ли система в бассейне текущего аттрактора.

    Для локального шума: если проигрыш не превышает порог, остаёмся.
    При history_size > 0 аттрактор сильнее (STP), порог эффективный растёт.

    Args:
        basin_threshold: ε — базовый порог бассейна притяжения.
        score_current: Оценка текущей задачи.
        score_runner_up: Оценка претендента.
        history_size: Текущая история удержания (влияет на STP).

    Returns:
        True, если переключение НЕ требуется (в бассейне).
    """
    # Чем больше history_size, тем сложнее выбить из аттрактора (STP)
    effective_threshold = basin_threshold + 0.05 * history_size
    delta = score_current - score_runner_up
    # Остаёмся, если проигрываем не более чем на эффективный порог
    return delta > -effective_threshold


def check_immediate_switch(
    score_current: float,
    score_runner_up: float,
    dominance_threshold: float = 0.3,
) -> bool:
    """Проверить, есть ли явное превосходство конкурента.

    При сильном проигрыше (Δ < -dominance_threshold) разрешаем немедленное
    переключение, игнорируя min_dwell — это не дребезг, а осознанная уступка.
    Отличается от dwell-механизма: dwell защищает от быстрых флип-флопов,
    а immediate_switch разрешает переключение, когда конкурент ЯВНО выигрывает.

    Args:
        score_current: Оценка текущей задачи.
        score_runner_up: Оценка претендента.
        dominance_threshold: Порог явного превосходства (Δscore < -ε).

    Returns:
        True, если конкурент явно выигрывает (переключаемся немедленно).
    """
    return (score_runner_up - score_current) > dominance_threshold
