"""Throttle — homeostatic regulation of the host's own parameters (S4).

A critical interoceptive signal (severity >= 0.9, ``is_critical``) triggers a
reflex: the host reduces its own computational load. This is internal and
reversible (never a world action) — the reflex path from manifest §3.К, which
bypasses policy, dwell and attractor.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.homeostasis import HomeostasisState


@dataclass(frozen=True)
class ThrottlePlan:
    """План регуляции собственных параметров хоста.

    Attributes:
        active: Активен ли throttle (есть критический канал).
        k_scale: Множитель k-WTA, (0, 1].
        dt_scale: Множитель шага dt, >= 1 (медленнее тики).
        llm_gate: Запретить инициативную речь (дорогой вызов LLM).
        reason: Причина (тег критического канала + severity).
    """

    active: bool
    k_scale: float = 1.0
    dt_scale: float = 1.0
    llm_gate: bool = False
    reason: str = ""


def plan_throttle(
    homeostasis: HomeostasisState,
    *,
    severity_threshold: float = 0.9,
    k_scale: float = 0.5,
    dt_scale: float = 2.0,
) -> ThrottlePlan:
    """Построить план throttle по снимку гомеостаза (чистая функция).

    Args:
        homeostasis: Снимок гомеостаза.
        severity_threshold: Порог критического сигнала, [0, 1].
        k_scale: Во сколько раз уменьшить k, (0, 1].
        dt_scale: Во сколько раз увеличить dt, >= 1.

    Returns:
        ThrottlePlan; ``active=False`` при отсутствии критических каналов.

    Raises:
        ValueError: Если severity_threshold вне [0, 1], k_scale вне (0, 1]
            или dt_scale < 1.
    """
    if not 0.0 <= severity_threshold <= 1.0:
        raise ValueError(
            f"severity_threshold must be in [0, 1], got {severity_threshold}"
        )
    if not 0.0 < k_scale <= 1.0:
        raise ValueError(f"k_scale must be in (0, 1], got {k_scale}")
    if dt_scale < 1.0:
        raise ValueError(f"dt_scale must be >= 1, got {dt_scale}")

    critical = [s for s in homeostasis.signals if s.severity >= severity_threshold]
    if not critical:
        return ThrottlePlan(active=False)

    worst = max(critical, key=lambda s: s.severity)
    return ThrottlePlan(
        active=True,
        k_scale=k_scale,
        dt_scale=dt_scale,
        llm_gate=True,
        reason=(
            f"critical {worst.tag}: severity={worst.severity:.2f} "
            f"-> throttle k×{k_scale}, dt×{dt_scale}"
        ),
    )
