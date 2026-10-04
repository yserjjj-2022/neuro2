"""Policy — action selection with causal traceability (S4).

Re-exports:
    Action — speech-action candidates
    Preferences — preferred outcomes (goal-directed)
    PolicyContext — extensible policy input
    PolicyCandidate — scored candidate
    PolicyTrace — causal trace of the decision
    evaluate_candidates — pure function: score all candidates
    select_action — pure function: deterministic choice + trace
"""

from src.core.policy.compute import evaluate_candidates, select_action
from src.core.policy.models import (
    Action,
    MacroContext,
    MetacognitionView,
    PartnerView,
    PolicyCandidate,
    PolicyContext,
    PolicyTrace,
    Preferences,
)

__all__ = [
    "Action",
    "MacroContext",
    "MetacognitionView",
    "PartnerView",
    "PolicyCandidate",
    "PolicyContext",
    "PolicyTrace",
    "Preferences",
    "evaluate_candidates",
    "select_action",
]
