"""Theory of Mind (S5) — partner model, recognition, Vigilance, Joint Agency.

Re-exports:
    PartnerState — ToM snapshot (trust/ambiguity/conflict/uncertainty/name)
    PartnerSignature — derived regularity (centroid + meta), not a tag
    match_partner — pure: match an utterance to accumulated signatures
    update_signature — pure: update/create a signature
    update_trust — pure: update trust/ambiguity/conflict
    PartnerModel — imperative shell owning signatures

Identity is a *derived regularity*, not a tag (ADR-0008). Signatures live in the
shared-memory domain; there is no separate store.
"""

from src.tm.compute import (
    detect_conflict,
    match_partner,
    normalize_pause,
    update_signature,
    update_trust,
)
from src.tm.joint import JointAgency
from src.tm.models import Claim, JointGoal, PartnerSignature, PartnerState
from src.tm.partner import PartnerModel
from src.tm.vigilance import VigilanceGate

__all__ = [
    "Claim",
    "JointAgency",
    "JointGoal",
    "PartnerModel",
    "PartnerSignature",
    "PartnerState",
    "VigilanceGate",
    "detect_conflict",
    "match_partner",
    "normalize_pause",
    "update_signature",
    "update_trust",
]
