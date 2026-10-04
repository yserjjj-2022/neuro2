"""Integration registry — catalog of the host's organs (ADR-0011).

- Catalog (config): ``IntegrationRegistry`` + ``default_integrations``.
- Loader: ``load_integrations`` (Python base + TOML override).
- Bridges (Core): ``to_provider`` (SENSOR), ``to_affordances`` (TOOL).

Runtime artifacts (``AffordanceMap``, ``SignalProvider``) are derived from the
catalog; the catalog is the single source of truth.
"""

from .bridges import to_affordances, to_provider
from .models import (
    HttpTransport,
    IntegrationKind,
    IntegrationSpec,
    LocalTransport,
    Provenance,
    StdioTransport,
    Transport,
)
from .registry import IntegrationRegistry, default_integrations
from .toml_loader import load_integrations

__all__ = [
    "HttpTransport",
    "IntegrationKind",
    "IntegrationRegistry",
    "IntegrationSpec",
    "LocalTransport",
    "Provenance",
    "StdioTransport",
    "Transport",
    "default_integrations",
    "load_integrations",
    "to_affordances",
    "to_provider",
]
