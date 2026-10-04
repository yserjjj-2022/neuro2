"""Bridges from the catalog to runtime artifacts (ADR-0011 §6, Core).

The registry is the single source of truth; runtime views are derived:
- ``to_provider`` — SENSOR → ``SignalProvider`` (bus u(t));
- ``to_affordances`` — TOOL → ``AffordanceMap`` (epistemic probing).

Pure functions: no I/O, deterministic. MCP sensors need a client-backed
provider and are not built here yet (raise ``NotImplementedError``).
"""

from __future__ import annotations

from collections.abc import Sequence

from src.host.sources import (
    CircadianProvider,
    SignalProvider,
)
from src.integrations.models import (
    IntegrationKind,
    IntegrationSpec,
    LocalTransport,
)
from src.mcp.probe import Affordance, AffordanceMap

# Встроенные провайдеры по имени (LocalTransport.provider → фабрика).
# ResourceProvider требует meter (инъекция) → строится отдельно в wiring.
_LOCAL_BUILTINS: dict[str, type[SignalProvider]] = {
    "circadian": CircadianProvider,
}


def to_provider(spec: IntegrationSpec) -> SignalProvider:
    """Построить сенсорный провайдер из записи каталога (Core).

    Args:
        spec: Запись интеграции вида SENSOR.

    Returns:
        SignalProvider для укладки на шину u(t).

    Raises:
        ValueError: Если spec не SENSOR.
        NotImplementedError: Для MCP-сенсоров (нужен клиент) или неизвестного
            локального провайдера.
    """
    if spec.kind is not IntegrationKind.SENSOR:
        raise ValueError(
            f"to_provider expects SENSOR, got {spec.kind.value!r} ({spec.name})"
        )
    if isinstance(spec.transport, LocalTransport):
        factory = _LOCAL_BUILTINS.get(spec.transport.provider)
        if factory is None:
            raise NotImplementedError(
                f"local provider {spec.transport.provider!r} not buildable here "
                f"(inject via wiring): {spec.name}"
            )
        return factory()
    raise NotImplementedError(
        f"MCP sensor provider not implemented yet: {spec.name}"
    )


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


__all__ = ["to_affordances", "to_provider"]
