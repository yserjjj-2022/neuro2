"""Factorization — sparse discrete layer over independent factors (S6).

Functional Core (ADR-0004): pure functions, no I/O. Independent small factors
(mode / partner / task) instead of a single combinatorial POMDP matrix
(manifest §3.Е). ``pymdp`` is an optional future; this module is NumPy-only
(ADR-0009 §6).

Re-exports:
    Factor — independent discrete factor (prior + likelihood)
    FactorizedState — product of independent factors
    posterior — Bayesian update of one factor
    marginal — factor distribution by name
    update_factor — isolate update of one factor
    argmax_state — MAP state of a factor
"""

from src.core.factorization.compute import (
    argmax_state,
    marginal,
    posterior,
    update_factor,
)
from src.core.factorization.models import Factor, FactorizedState

__all__ = [
    "Factor",
    "FactorizedState",
    "argmax_state",
    "marginal",
    "posterior",
    "update_factor",
]
