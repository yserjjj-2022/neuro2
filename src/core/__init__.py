"""Core modules: CMC fabric, energy, voting, attractors.

CMC (Canonical Microcircuits): L4 → L5/6 → L2/3 column dynamics.
Voting: k-WTA lateral inhibition for column consensus.
Attractors: multistable dynamics for task selection via short-term plasticity.
Functional Core / Imperative Shell (ADR-0004) across all core/* modules.
"""

from .attractors import (
    TaskAttraction,
    TaskAttractor,
    check_basin_stability,
    check_immediate_switch,
    compute_dwell,
)
from .cmc import CMCEnsemble, ColumnConfig, ColumnState, EnsembleOutput, column_step
from .voting import VotingManager, VotingResult, kwta

__all__ = [
    "CMCEnsemble",
    "ColumnConfig",
    "ColumnState",
    "EnsembleOutput",
    "TaskAttraction",
    "TaskAttractor",
    "VotingManager",
    "VotingResult",
    "check_basin_stability",
    "check_immediate_switch",
    "column_step",
    "compute_dwell",
    "kwta",
]
