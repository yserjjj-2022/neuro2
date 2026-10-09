"""Core modules: CMC fabric, energy, voting, attractors, homeostasis.

CMC (Canonical Microcircuits): L4 → L5/6 → L2/3 column dynamics.
Voting: k-WTA lateral inhibition for column consensus.
Attractors: multistable dynamics for task selection via short-term plasticity.
Homeostasis: interoceptive setpoints and deviation (S4).
Actuation: open option window over tools for goal selection (S8).
Functional Core / Imperative Shell (ADR-0004) across all core/* modules.
"""

from .actuation import (
    ActuationPreferences,
    Option,
    OptionCandidate,
    OptionContext,
    OptionSource,
    OptionTrace,
    OptionWindow,
    build_options,
    score_option,
    select_option,
)
from .attractors import (
    TaskAttraction,
    TaskAttractor,
    check_basin_stability,
    check_immediate_switch,
    compute_dwell,
)
from .cmc import CMCEnsemble, ColumnConfig, ColumnState, EnsembleOutput, column_step
from .homeostasis import (
    HomeostasisState,
    Homeostat,
    HomeostaticSignal,
    Setpoint,
    setpoint_deviation,
)
from .policy import (
    Action,
    MacroContext,
    PolicyCandidate,
    PolicyContext,
    PolicyTrace,
    Preferences,
    evaluate_candidates,
    select_action,
)
from .voting import VotingManager, VotingResult, kwta

__all__ = [
    "Action",
    "ActuationPreferences",
    "CMCEnsemble",
    "ColumnConfig",
    "ColumnState",
    "EnsembleOutput",
    "HomeostasisState",
    "Homeostat",
    "HomeostaticSignal",
    "MacroContext",
    "Option",
    "OptionCandidate",
    "OptionContext",
    "OptionSource",
    "OptionTrace",
    "OptionWindow",
    "PolicyCandidate",
    "PolicyContext",
    "PolicyTrace",
    "Preferences",
    "Setpoint",
    "TaskAttraction",
    "TaskAttractor",
    "VotingManager",
    "VotingResult",
    "build_options",
    "check_basin_stability",
    "check_immediate_switch",
    "column_step",
    "compute_dwell",
    "evaluate_candidates",
    "kwta",
    "score_option",
    "select_action",
    "select_option",
    "setpoint_deviation",
]
