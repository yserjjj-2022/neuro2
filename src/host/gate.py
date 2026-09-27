"""Capability gate — single point of side-effect authorization (S4).

Manifest §3.И / ADR-0005 §9: every action with a side effect passes through
one gate that checks tier, reversibility and HITL. Fail-safe deny: unknown
tier, missing HITL token or timeout → refusal (never a silent allow).

On S4 the host has no external side effects (speech is reversible, throttle is
internal), so the gate is a *skeleton*: it is exercised and audited, but does
not yet block anything irreversible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CapabilityTier(Enum):
    """Уровни возможностей хоста (ADR-0005 §6).

    T0 наблюдение · T1 речь · T2 действие с HITL · T3 автономное обратимое ·
    T4 ограниченное необратимое.
    """

    T0 = "observation"
    T1 = "speech"
    T2 = "hitl_action"
    T3 = "autonomous_reversible"
    T4 = "bounded_irreversible"


# Числовой ранг tier для сравнения (T0 < T1 < ... < T4).
_TIER_RANK: dict[CapabilityTier, int] = {
    CapabilityTier.T0: 0,
    CapabilityTier.T1: 1,
    CapabilityTier.T2: 2,
    CapabilityTier.T3: 3,
    CapabilityTier.T4: 4,
}


@dataclass(frozen=True)
class ActionRequest:
    """Запрос на действие через gate.

    Attributes:
        name: Имя действия ("speak", "throttle", ...).
        tier: Требуемый уровень возможностей.
        reversible: Обратимо ли действие.
        reason: Причина (из policy trace / throttle plan) — для аудита.
        hitl_token: Токен одобрения человека (обязателен для необратимых).
    """

    name: str
    tier: CapabilityTier
    reversible: bool
    reason: str
    hitl_token: str | None = None


@dataclass(frozen=True)
class GateDecision:
    """Решение gate.

    Attributes:
        allowed: Разрешено ли действие.
        reason: Причина решения (для аудита).
    """

    allowed: bool
    reason: str


class CapabilityGate:
    """Единая точка проверки действий перед side-effect.

    Правила (fail-safe deny):
        - tier ≤ max_tier → разрешено;
        - tier > max_tier → отказ;
        - необратимое действие требует HITL-токена, иначе отказ
          (таймаут/отсутствие → deny, не allow).

    Attributes:
        max_tier: Максимально разрешённый tier.
    """

    def __init__(self, max_tier: CapabilityTier = CapabilityTier.T1) -> None:
        """Создать gate.

        Args:
            max_tier: Максимально разрешённый tier (дефолт T1 — речь).
        """
        self.max_tier = max_tier

    def request(self, req: ActionRequest) -> GateDecision:
        """Проверить запрос действия.

        Args:
            req: Запрос на действие.

        Returns:
            GateDecision — разрешение/отказ с причиной (аудит).
        """
        if _TIER_RANK[req.tier] > _TIER_RANK[self.max_tier]:
            decision = GateDecision(
                allowed=False,
                reason=(
                    f"denied {req.name}: tier {req.tier.value} > "
                    f"max {self.max_tier.value}"
                ),
            )
        elif not req.reversible and not req.hitl_token:
            decision = GateDecision(
                allowed=False,
                reason=f"denied {req.name}: irreversible without HITL token",
            )
        else:
            decision = GateDecision(
                allowed=True, reason=f"allowed {req.name} (tier {req.tier.value})"
            )

        logger.debug("gate: %s | request=%s", decision.reason, req.name)
        return decision
