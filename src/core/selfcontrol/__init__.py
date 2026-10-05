"""Selfcontrol — metacognition, drift classification and reset protocol (S6).

Functional Core / Imperative Shell (ADR-0004):
- Core (compute.py) — pure functions, tested without Shell
- Shell (monitor.py) — SelfMonitor keeps ring buffers, computes observables

Re-exports:
    Metacognition — read-only metacognitive snapshot (policy input)
    CriticalSlowingDown — early-warning signal (variance + autocorrelation)
    ChangeKind, ChangeAssessment — development vs drift classification
    ResetLevel, ResetPlan — reset protocol (soft/freeze/hard)
    ResetReport, reset_self_report — honest self-report of a reset (S7)
    SelfMonitor — imperative shell over the pure core
    compute_conflict, compute_metastability, compute_saturation — observables
    critical_slowing_down — pure early-warning estimator
    classify_change — pure development/drift classifier
    plan_reset — pure reset planner
"""

from src.core.selfcontrol.compute import (
    classify_change,
    compute_conflict,
    compute_metastability,
    compute_saturation,
    critical_slowing_down,
    plan_reset,
)
from src.core.selfcontrol.models import (
    ChangeAssessment,
    ChangeKind,
    CriticalSlowingDown,
    Metacognition,
    ResetLevel,
    ResetPlan,
)
from src.core.selfcontrol.monitor import SelfMonitor
from src.core.selfcontrol.report import ResetReport, reset_self_report

__all__ = [
    "ChangeAssessment",
    "ChangeKind",
    "CriticalSlowingDown",
    "Metacognition",
    "ResetLevel",
    "ResetPlan",
    "ResetReport",
    "SelfMonitor",
    "classify_change",
    "compute_conflict",
    "compute_metastability",
    "compute_saturation",
    "critical_slowing_down",
    "plan_reset",
    "reset_self_report",
]
