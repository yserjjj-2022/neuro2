"""Actuation sequencing core (S8 stages 1–5, ADR-0012).

Open option window + total scorer + deterministic choice (stage 1); facts and
conditions "default + enrichment" — ``Fact``/``Guard``/``Regularity`` with
``evaluate_fact``/``guard_holds``/``regularity_cost`` (stage 2); effects and
irreversibility as a stance — ``Effect``/``Actuation``/``ActuationResult``/
``ToolAnnotations`` with ``classify_reversible`` (stage 3); reactive Behavior
Tree — ``Node``/``NodeKind``/``NodeStatus`` with ``tick``/``order_children``
(stage 4); runtime generator — ``Goal`` with ``backward_chain`` (stage 5).
Pure Functional Core (ADR-0004).
"""

from .compute import (
    backward_chain,
    build_options,
    classify_reversible,
    evaluate_fact,
    guard_holds,
    order_children,
    regularity_cost,
    score_option,
    select_option,
    tick,
)
from .facts import FACTS, NETWORK_AVAILABLE, TOPIC_BOUND
from .models import (
    DEFAULT_FACT_VALUE,
    Actuation,
    ActuationKind,
    ActuationPreferences,
    ActuationResult,
    ActuationStatus,
    Effect,
    Fact,
    Goal,
    Guard,
    Node,
    NodeKind,
    NodeStatus,
    Option,
    OptionCandidate,
    OptionContext,
    OptionSource,
    OptionTrace,
    OptionWindow,
    Regularity,
    TickContext,
    TickMemory,
    ToolAnnotations,
)

__all__ = [
    "DEFAULT_FACT_VALUE",
    "FACTS",
    "NETWORK_AVAILABLE",
    "TOPIC_BOUND",
    "Actuation",
    "ActuationKind",
    "ActuationPreferences",
    "ActuationResult",
    "ActuationStatus",
    "Effect",
    "Fact",
    "Goal",
    "Guard",
    "Node",
    "NodeKind",
    "NodeStatus",
    "Option",
    "OptionCandidate",
    "OptionContext",
    "OptionSource",
    "OptionTrace",
    "OptionWindow",
    "Regularity",
    "TickContext",
    "TickMemory",
    "ToolAnnotations",
    "backward_chain",
    "build_options",
    "classify_reversible",
    "evaluate_fact",
    "guard_holds",
    "order_children",
    "regularity_cost",
    "score_option",
    "select_option",
    "tick",
]
