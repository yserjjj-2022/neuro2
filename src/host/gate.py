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


class Capability(Enum):
    """Гранулярные права действия (S4-долг: развести чтение/речь/действие).

    Tier отвечает на вопрос «насколько далеко» действие (для HITL/аудита),
    capability — «какое именно право» нужно. Разведение позволяет, например,
    сохранить хосту право ``SPEAK`` под throttle (escape hatch), блокируя
    дорогое ``ACT_REVERSIBLE``. Права независимы от tier-лестницы.
    """

    READ = "read"
    THINK = "think"
    SPEAK = "speak"
    ACT_REVERSIBLE = "act_reversible"
    ACT_IRREVERSIBLE = "act_irreversible"


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
        tier: Требуемый уровень возможностей (для HITL/аудита).
        reversible: Обратимо ли действие.
        reason: Причина (из policy trace / throttle plan) — для аудита.
        hitl_token: Токен одобрения человека (обязателен для необратимых).
        capabilities: Гранулярные права, требуемые действием (S4-долг).
            Пусто → действие не требует отдельных прав (совместимость).
    """

    name: str
    tier: CapabilityTier
    reversible: bool
    reason: str
    hitl_token: str | None = None
    capabilities: frozenset[Capability] = frozenset()


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

    Правила (fail-safe deny), в порядке проверки:
        1. требуемые capabilities ⊆ granted → иначе отказ (право отозвано);
        2. tier ≤ max_tier → иначе отказ;
        3. необратимое действие требует HITL-токена, иначе отказ
           (таймаут/отсутствие → deny, не allow).

    Разведение прав (capabilities) и tier позволяет сохранить хосту
    «право подать голос» под стрессом, отозвав дорогое право (напр.
    ``THINK`` — инициативный вызов LLM), вместо глухой блокировки.

    Attributes:
        max_tier: Максимально разрешённый tier.
        granted: Набор разрешённых прав (frozenset).
    """

    def __init__(
        self,
        max_tier: CapabilityTier = CapabilityTier.T1,
        *,
        granted: frozenset[Capability] | None = None,
    ) -> None:
        """Создать gate.

        Args:
            max_tier: Максимально разрешённый tier (дефолт T1 — речь).
            granted: Разрешённые права; None → все права разрешены.
        """
        self.max_tier = max_tier
        self._granted: frozenset[Capability] = (
            frozenset(Capability) if granted is None else frozenset(granted)
        )

    @property
    def granted(self) -> frozenset[Capability]:
        """Текущий набор разрешённых прав (read-only view)."""
        return self._granted

    def grant(self, capability: Capability) -> None:
        """Разрешить право.

        Args:
            capability: Право для выдачи.
        """
        self._granted = self._granted | {capability}

    def revoke(self, capability: Capability) -> None:
        """Отозвать право (fail-safe: действие, требующее его, будет отклонено).

        Args:
            capability: Право для отзыва.
        """
        self._granted = self._granted - {capability}

    def request(self, req: ActionRequest) -> GateDecision:
        """Проверить запрос действия.

        Args:
            req: Запрос на действие.

        Returns:
            GateDecision — разрешение/отказ с причиной (аудит).
        """
        missing = req.capabilities - self._granted
        if missing:
            names = ",".join(sorted(c.value for c in missing))
            decision = GateDecision(
                allowed=False,
                reason=f"denied {req.name}: missing capabilities [{names}]",
            )
        elif _TIER_RANK[req.tier] > _TIER_RANK[self.max_tier]:
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
