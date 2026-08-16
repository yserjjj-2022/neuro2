"""Core modules: CMC fabric, energy, voting.

CMC (Canonical Microcircuits): L4 → L5/6 → L2/3 column dynamics.
Functional Core / Imperative Shell (ADR-0004) across all core/* modules.
"""

from .cmc import CMCEnsemble, ColumnConfig, ColumnState, EnsembleOutput, column_step

__all__ = [
    "CMCEnsemble",
    "ColumnConfig",
    "ColumnState",
    "EnsembleOutput",
    "column_step",
]
