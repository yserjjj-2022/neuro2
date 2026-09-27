"""Functional Core — pure functions for homeostasis.

Functional Core / Imperative Shell (ADR-0004): identical inputs → identical
outputs, no mutation, no I/O. Deviation is the normalized distance of a raw
channel severity from its setpoint.
"""

from __future__ import annotations


def setpoint_deviation(
    severity: float,
    comfort: float,
    critical: float,
) -> float:
    """Нормированное отклонение severity от сетпоинта.

    Формула:
        deviation = clip((severity - comfort) / (critical - comfort), 0, 1)

    Семантика:
        - severity <= comfort → 0.0 (канал в норме).
        - severity == critical → 1.0 (максимальное отклонение; совпадает
          с порогом reflex-пути).
        - Линейно между comfort и critical.

    Args:
        severity: Сырая severity канала, [0, 1].
        comfort: Верхняя граница нормы.
        critical: Severity, при которой deviation == 1.0.

    Returns:
        Отклонение ∈ [0, 1].

    Raises:
        ValueError: Если critical <= comfort.

    Examples:
        >>> setpoint_deviation(0.5, 0.5, 0.9)
        0.0
        >>> setpoint_deviation(0.9, 0.5, 0.9)
        1.0
        >>> setpoint_deviation(0.7, 0.5, 0.9)
        0.5
    """
    if critical <= comfort:
        raise ValueError(f"critical must be > comfort, got {critical} <= {comfort}")
    raw = (severity - comfort) / (critical - comfort)
    return float(min(1.0, max(0.0, raw)))
