"""MCP — Model Context Protocol integration.

Signal taxonomy: SignalCategory enum + SignalSource frozen dataclass.
Registry: SignalRegistry — manages sources, aggregates into unified vector.

Functional Core / Imperative Shell (ADR-0004):
- Core (models) — pure dataclasses, no I/O
- Shell (registry) — owns sources list, aggregates
"""

from .models import SignalCategory, SignalSource
from .registry import SignalRegistry

__all__ = ["SignalCategory", "SignalSource", "SignalRegistry"]
