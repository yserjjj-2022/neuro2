"""Actuation sequencing core (S8 stages 1–3, ADR-0012).

Open option window + total scorer + deterministic choice (stage 1); facts and
conditions "default + enrichment" — ``Fact``/``Guard``/``Regularity`` with
``evaluate_fact``/``guard_holds``/``regularity_cost`` (stage 2); effects and
irreversibility as a stance — ``Effect``/``Actuation``/``ActuationResult``/
``ToolAnnotations`` with ``classify_reversible`` (stage 3). Pure Functional
Core (ADR-0004).
"""

from .compute import (
    build_options,
    classify_reversible,
    evaluate_fact,
    guard_holds,
    regularity_cost,
    score_option,
    select_option,
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
    Guard,
    Option,
    OptionCandidate,
    OptionContext,
    OptionSource,
    OptionTrace,
    OptionWindow,
    Regularity,
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
    "Guard",
    "Option",
    "OptionCandidate",
    "OptionContext",
    "OptionSource",
    "OptionTrace",
    "OptionWindow",
    "Regularity",
    "ToolAnnotations",
    "build_options",
    "classify_reversible",
    "evaluate_fact",
    "guard_holds",
    "regularity_cost",
    "score_option",
    "select_option",
]
