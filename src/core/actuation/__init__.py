"""Actuation sequencing core (S8 stages 1–2, ADR-0012).

Open option window + total scorer + deterministic choice (stage 1); facts and
conditions "default + enrichment" — ``Fact``/``Guard``/``Regularity`` with
``evaluate_fact``/``guard_holds``/``regularity_cost`` (stage 2). Pure Functional
Core (ADR-0004).
"""

from .compute import (
    build_options,
    evaluate_fact,
    guard_holds,
    regularity_cost,
    score_option,
    select_option,
)
from .facts import FACTS, NETWORK_AVAILABLE, TOPIC_BOUND
from .models import (
    DEFAULT_FACT_VALUE,
    ActuationPreferences,
    Fact,
    Guard,
    Option,
    OptionCandidate,
    OptionContext,
    OptionSource,
    OptionTrace,
    OptionWindow,
    Regularity,
)

__all__ = [
    "DEFAULT_FACT_VALUE",
    "FACTS",
    "NETWORK_AVAILABLE",
    "TOPIC_BOUND",
    "ActuationPreferences",
    "Fact",
    "Guard",
    "Option",
    "OptionCandidate",
    "OptionContext",
    "OptionSource",
    "OptionTrace",
    "OptionWindow",
    "Regularity",
    "build_options",
    "evaluate_fact",
    "guard_holds",
    "regularity_cost",
    "score_option",
    "select_option",
]
