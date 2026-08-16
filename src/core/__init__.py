"""Core modules: CMC fabric, energy, voting.

CMC (Canonical Microcircuits): L4 → L5/6 → L2/3 column dynamics.
Voting: k-WTA lateral inhibition for column consensus.
Functional Core / Imperative Shell (ADR-0004) across all core/* modules.
"""

from .cmc import CMCEnsemble, ColumnConfig, ColumnState, EnsembleOutput, column_step
from .voting import VotingManager, VotingResult, kwta

__all__ = [
    "CMCEnsemble",
    "ColumnConfig",
    "ColumnState",
    "EnsembleOutput",
    "VotingManager",
    "VotingResult",
    "column_step",
    "kwta",
]
