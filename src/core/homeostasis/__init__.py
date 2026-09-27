"""Homeostasis — interoceptive setpoints and deviation (S4).

Re-exports:
    Setpoint — channel setpoint (comfort/critical/weight)
    HomeostaticSignal — per-channel evaluation
    HomeostasisState — aggregate snapshot
    setpoint_deviation — pure function: normalized deviation
    Homeostat — imperative shell: evaluates bus signals
"""

from src.core.homeostasis.compute import setpoint_deviation
from src.core.homeostasis.manager import Homeostat
from src.core.homeostasis.models import (
    HomeostasisState,
    HomeostaticSignal,
    Setpoint,
)

__all__ = [
    "HomeostasisState",
    "Homeostat",
    "HomeostaticSignal",
    "Setpoint",
    "setpoint_deviation",
]
