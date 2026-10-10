"""Bridges from the catalog to runtime artifacts (ADR-0011 §6, Core).

The registry is the single source of truth; runtime views are derived:
- ``to_affordances`` — TOOL → ``AffordanceMap`` (epistemic probing).

SENSOR → ``SignalProvider`` assembly moved to ``factories.py`` (Shell-glue):
it needs injected dependencies (meter, embedder) and must not drag host/memory
types into this pure Core module.

Pure functions: no I/O, deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.integrations.models import (
    IntegrationKind,
    IntegrationSpec,
)
from src.mcp.probe import Affordance, AffordanceMap


def to_affordances(specs: Sequence[IntegrationSpec]) -> AffordanceMap:
    """Построить карту аффордансов из TOOL-записей каталога (Core).

    Для каждой записи: если ``tools`` задан — по одному аффордансу на имя,
    иначе — один аффорданс с именем интеграции. Не-TOOL записи игнорируются;
    выключенные записи пропускаются.

    Args:
        specs: Записи каталога.

    Returns:
        AffordanceMap с уникальными именами в порядке записей.

    Raises:
        ValueError: При дублирующихся именах аффордансов (fail-fast).
    """
    affordances: list[Affordance] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.kind is not IntegrationKind.TOOL or not spec.enabled:
            continue
        names = spec.tools if spec.tools else (spec.name,)
        for tool_name in names:
            if tool_name in seen:
                raise ValueError(f"duplicate affordance name: {tool_name!r}")
            seen.add(tool_name)
            affordances.append(
                Affordance(
                    name=tool_name,
                    category=spec.category,
                    reversible=spec.reversible,
                )
            )
    return AffordanceMap(tuple(affordances))


__all__ = ["to_affordances"]
