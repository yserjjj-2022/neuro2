"""Task attractors module — multistable dynamics for task selection.

Re-exports:
    TaskAttraction — frozen dataclass, snapshot of attractor state
    TaskAttractor — imperative shell, manages attractor dynamics
    compute_dwell — pure function: compute dwell time (hysteresis + STP)
    check_basin_stability — pure function: basin of attraction check
    check_immediate_switch — pure function: explicit competitor dominance
"""

from src.core.attractors.compute import (
    check_basin_stability,
    check_immediate_switch,
    compute_dwell,
)
from src.core.attractors.manager import TaskAttractor
from src.core.attractors.models import TaskAttraction

__all__ = [
    "TaskAttraction",
    "TaskAttractor",
    "check_basin_stability",
    "check_immediate_switch",
    "compute_dwell",
]
