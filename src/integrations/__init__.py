"""Integration registry — catalog of the host's organs (ADR-0011).

- Catalog (config): ``IntegrationRegistry`` + ``default_integrations``.
- Loader: ``load_integrations`` (Python base + TOML override).
- Bridges (Core): ``to_affordances`` (TOOL → AffordanceMap).
- Factories (Shell): ``build_provider``/``build_providers`` + ``SensorContext``
  (SENSOR → SignalProvider, with injected deps).

Runtime artifacts (``AffordanceMap``, ``SignalProvider``) are derived from the
catalog; the catalog is the single source of truth.
"""

from .bridges import to_affordances
from .factories import (
    SensorContext,
    build_provider,
    build_providers,
    enabled_sensors,
)
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
    "SensorContext",
    "StdioTransport",
    "Transport",
    "build_provider",
    "build_providers",
    "default_integrations",
    "enabled_sensors",
    "load_integrations",
    "to_affordances",
]
