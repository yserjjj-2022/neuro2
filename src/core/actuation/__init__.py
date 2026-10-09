"""Actuation sequencing core (S8 stage 1, ADR-0012).

Open option window + total scorer + deterministic choice. Pure Functional
Core (ADR-0004): ``build_options`` / ``score_option`` / ``select_option``.
"""

from .compute import build_options, score_option, select_option
from .models import (
    ActuationPreferences,
    Option,
    OptionCandidate,
    OptionContext,
    OptionSource,
    OptionTrace,
    OptionWindow,
)

__all__ = [
    "ActuationPreferences",
    "Option",
    "OptionCandidate",
    "OptionContext",
    "OptionSource",
    "OptionTrace",
    "OptionWindow",
    "build_options",
    "score_option",
    "select_option",
]
