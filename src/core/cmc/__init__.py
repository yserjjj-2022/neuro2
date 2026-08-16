"""Canonical Microcircuits (CMC) — column dynamics and ensemble aggregation.

Layers: L4 (input) → L5/6 (state x(t)) → L2/3 (error e(t)).
Functional Core / Imperative Shell (ADR-0004):
- column_step — pure function, state passed explicitly
- CMCEnsemble — shell owning column states, aggregates per tick
"""

from .column import column_step
from .ensemble import CMCEnsemble
from .models import ColumnConfig, ColumnState, EnsembleOutput

__all__ = [
    "CMCEnsemble",
    "ColumnConfig",
    "ColumnState",
    "EnsembleOutput",
    "column_step",
]
