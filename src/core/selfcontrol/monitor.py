"""SelfMonitor — Imperative Shell over the pure selfcontrol core (S6).

Keeps fixed-length ring buffers of F and attractor-switch flags, computes the
metacognitive observables and the reset plan each tick. The monitor is a
read-only observer: it never mutates the affective loop and never influences
F(t) of the same tick (ADR-0009 §1).
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]

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
    Metacognition,
    ResetPlan,
)

logger = logging.getLogger(__name__)


class SelfMonitor:
    """Shell: копит окно метрик, считает наблюдаемые и план сброса.

    Attributes:
        window: Длина окна (тики) для наблюдаемых и CSD.
        variance_gain: Вес дисперсионной компоненты CSD.
        autocorr_gain: Вес автокорреляционной компоненты CSD.
        warning_threshold: Порог slowing для ``is_warning``.
        soft_threshold: Порог slowing для SOFT-сброса.
    """

    def __init__(
        self,
        *,
        window: int = 50,
        variance_gain: float = 1.0,
        autocorr_gain: float = 1.0,
        warning_threshold: float = 0.6,
        soft_threshold: float = 0.5,
    ) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if not 0.0 <= warning_threshold <= 1.0:
            raise ValueError(
                f"warning_threshold must be in [0, 1], got {warning_threshold}"
            )
        if not 0.0 <= soft_threshold <= 1.0:
            raise ValueError(
                f"soft_threshold must be in [0, 1], got {soft_threshold}"
            )
        self.window = window
        self.variance_gain = variance_gain
        self.autocorr_gain = autocorr_gain
        self.warning_threshold = warning_threshold
        self.soft_threshold = soft_threshold
        self._f_history: deque[float] = deque(maxlen=window)
        self._switch_flags: deque[bool] = deque(maxlen=window)
        self._last_metacognition: Metacognition | None = None
        self._last_assessment: ChangeAssessment | None = None

    @property
    def metacognition(self) -> Metacognition | None:
        """Последний снимок метакогниции (None до первого observe)."""
        return self._last_metacognition

    @property
    def last_assessment(self) -> ChangeAssessment | None:
        """Последняя классификация изменения (None до первого observe)."""
        return self._last_assessment

    def observe(
        self,
        *,
        scores: Vector | Sequence[float],
        switched: bool,
        f: float,
        partner_uncertainty: float,
        core_preserved: bool = True,
        traceable: bool = True,
        coherent: bool = True,
    ) -> tuple[Metacognition, ResetPlan]:
        """Обновить окна и вернуть наблюдаемые + план сброса.

        Args:
            scores: Активности колонок текущего тика.
            switched: Сменился ли аттрактор на этом тике.
            f: Текущее F(t).
            partner_uncertainty: Неопределённость идентичности партнёра.
            core_preserved: Ядро сохранено (детектор дрейфа; по умолчанию да).
            traceable: Изменение трассируемо (по умолчанию да).
            coherent: Траектория когерентна (по умолчанию да).

        Returns:
            (Metacognition, ResetPlan). При сбое расчёта — безопасный дефолт
            (нулевые наблюдаемые, не триггерящий план).
        """
        self._f_history.append(float(f))
        self._switch_flags.append(bool(switched))
        try:
            metacognition = Metacognition(
                conflict=compute_conflict(scores),
                metastability=compute_metastability(
                    list(self._switch_flags), window=self.window
                ),
                epistemic_uncertainty=float(
                    min(1.0, max(0.0, partner_uncertainty))
                ),
                saturation=compute_saturation(list(self._f_history)),
            )
        except (ValueError, TypeError) as exc:
            logger.error("selfcontrol: metacognition failed (%s)", exc)
            metacognition = Metacognition(0.0, 0.0, 0.0, 0.0)

        slowing = critical_slowing_down(
            list(self._f_history),
            variance_gain=self.variance_gain,
            autocorr_gain=self.autocorr_gain,
        )
        slowing = replace(slowing, is_warning=slowing.slowing >= self.warning_threshold)

        changed = self._is_changing()
        assessment = classify_change(
            core_preserved=core_preserved,
            traceable=traceable,
            coherent=coherent,
            changed=changed,
        )
        plan = plan_reset(
            slowing=slowing,
            assessment=assessment,
            soft_threshold=self.soft_threshold,
            hard_core_broken=not core_preserved,
        )
        self._last_metacognition = metacognition
        self._last_assessment = assessment
        return metacognition, plan

    def _is_changing(self) -> bool:
        """Есть ли изменение: сменился аттрактор в окне или F дрейфует."""
        if any(self._switch_flags):
            return True
        if len(self._f_history) < 2:
            return False
        first, last = self._f_history[0], self._f_history[-1]
        scale = abs(first) + 1.0
        return abs(last - first) / scale > 0.1
