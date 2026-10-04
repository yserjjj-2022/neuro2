"""MCP probe effector — gated epistemic probing (S6 проход 2, Shell).

Takes a pure selection (``src/mcp/probe.py``) and executes it through the
capability gate (ADR-0005 §9): the single point of side-effect authorization.
Fail-safe deny: an irreversible probe without a HITL token is refused, and a
missing/denied affordance is refused — never a silent allow.

The MCP transport is not implemented; the effector calls an injected ``probe_fn``
(mock in tests, real MCP client later) behind the gate, so the host loop never
changes when the mock is swapped for a real integration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from src.host.gate import (
    ActionRequest,
    Capability,
    CapabilityGate,
    CapabilityTier,
)
from src.mcp.probe import (
    Affordance,
    AffordanceMap,
    ProbeRequest,
    ProbeResult,
)

logger = logging.getLogger(__name__)

# Транспорт зондирования: имя аффорданса → вектор данных.
ProbeFn = Callable[[Affordance], tuple[float, ...]]


def _mock_probe(affordance: Affordance) -> tuple[float, ...]:
    """Детерминированный mock-транспорт (до реального MCP-клиента)."""
    return tuple(0.0 for _ in range(affordance.dim))


class ProbeEffector:
    """Shell: исполняет зондирование через capability gate.

    Attributes:
        affordances: Карта доступных аффордансов.
        gate: Capability gate (единая точка side-effect).
        probe_fn: Транспорт (инъекция; mock по умолчанию).
    """

    def __init__(
        self,
        *,
        affordances: AffordanceMap | None = None,
        gate: CapabilityGate | None = None,
        probe_fn: ProbeFn | None = None,
    ) -> None:
        self.affordances = (
            affordances if affordances is not None else AffordanceMap()
        )
        self.gate = gate if gate is not None else CapabilityGate()
        self.probe_fn: ProbeFn = probe_fn if probe_fn is not None else _mock_probe

    def probe(
        self,
        request: ProbeRequest,
        *,
        hitl_token: str | None = None,
    ) -> ProbeResult:
        """Выполнить зондирование, если gate разрешает.

        Args:
            request: Запрос (имя аффорданса + причина).
            hitl_token: Токен одобрения (обязателен для необратимых).

        Returns:
            ProbeResult; ``success=False`` при отказе/сбое (никогда не бросает
            наружу — сбой зондирования не должен ронять тик).
        """
        affordance = self.affordances.find(request.affordance)
        if affordance is None:
            return ProbeResult(
                request.affordance,
                False,
                (),
                f"unknown affordance {request.affordance!r}",
            )

        required = {Capability.READ}
        required.add(
            Capability.ACT_REVERSIBLE
            if affordance.reversible
            else Capability.ACT_IRREVERSIBLE
        )
        decision = self.gate.request(
            ActionRequest(
                name=f"probe:{affordance.name}",
                tier=CapabilityTier.T3 if affordance.reversible else CapabilityTier.T4,
                reversible=affordance.reversible,
                reason=request.reason,
                hitl_token=hitl_token,
                capabilities=frozenset(required),
            )
        )
        if not decision.allowed:
            logger.info("probe denied: %s", decision.reason)
            return ProbeResult(affordance.name, False, (), decision.reason)

        try:
            data = self.probe_fn(affordance)
        except Exception as exc:  # noqa: BLE001 — зондирование не роняет тик
            logger.error("probe failed: %s (%s)", affordance.name, exc)
            return ProbeResult(affordance.name, False, (), f"probe error: {exc}")

        return ProbeResult(affordance.name, True, tuple(data), decision.reason)
