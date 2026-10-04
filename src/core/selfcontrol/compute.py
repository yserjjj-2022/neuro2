"""Functional Core — pure metacognition, CSD, classification and reset planning.

Deterministic, side-effect-free. No LLM reasoning (ADR-0007): the reset trigger
is critical slowing down plus the change classifier, not a raw stress threshold
(ADR-0009 §2–3). Identical inputs → identical outputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]

from .models import (
    ChangeAssessment,
    ChangeKind,
    CriticalSlowingDown,
    ResetLevel,
    ResetPlan,
)

_EPS = 1e-12


def _clip01(value: float) -> float:
    """Зажать значение в [0, 1]."""
    return float(min(1.0, max(0.0, value)))


def _sigmoid(x: float) -> float:
    """Логистическая функция (устойчивая к переполнению)."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def compute_conflict(scores: Vector | Sequence[float]) -> float:
    """Несогласие ансамбля: нормированный разброс активностей, [0, 1].

    Энтропия распределения ``scores``, нормированная на ``log(n)``: два
    равных пика → высокий конфликт, один доминирующий → низкий. Пустой вход
    или один элемент → 0.0.

    Args:
        scores: Активности колонок (неотрицательные).

    Returns:
        Конфликт ∈ [0, 1].

    Raises:
        ValueError: Если есть отрицательная активность.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    if float(np.min(arr)) < 0.0:
        raise ValueError("scores must be non-negative")
    total = float(np.sum(arr))
    if total <= _EPS:
        return 0.0
    probs = arr / total
    nonzero = probs[probs > 0.0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    norm = math.log(arr.size)
    if norm <= _EPS:
        return 0.0
    return _clip01(entropy / norm)


def compute_metastability(
    switch_flags: Sequence[bool],
    *,
    window: int,
) -> float:
    """Частота смен аттрактора в окне, [0, 1].

    Args:
        switch_flags: Флаги смены аттрактора (последние ``window`` тиков).
        window: Длина окна (>= 1).

    Returns:
        Доля смен ∈ [0, 1].

    Raises:
        ValueError: Если ``window < 1``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    recent = list(switch_flags)[-window:]
    if not recent:
        return 0.0
    switches = sum(1 for flag in recent if flag)
    return _clip01(switches / len(recent))


def compute_saturation(
    f_trend: Sequence[float],
    *,
    threshold: float | None = None,
) -> float:
    """Насыщение/тренд F: доля тиков выше порога, [0, 1].

    Если порог не задан — используется адаптивный порог ``mean + std``:
    хронизация нагрузки (стабильно высокий F) даёт рост насыщения.

    Args:
        f_trend: История F(t) (последние тики).
        threshold: Порог F (None → mean + std ряда).

    Returns:
        Насыщение ∈ [0, 1].
    """
    arr = np.asarray(f_trend, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if threshold is None:
        threshold = float(np.mean(arr) + np.std(arr))
    if arr.size == 1:
        return 1.0 if float(arr[0]) > threshold else 0.0
    above = float(np.mean(arr > threshold))
    return _clip01(above)


def _lag1_autocorrelation(series: np.ndarray) -> float:
    """Lag-1 автокорреляция ряда (0.0 при нулевой дисперсии)."""
    if series.size < 2:
        return 0.0
    x = series[:-1]
    y = series[1:]
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom <= _EPS:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def critical_slowing_down(
    series: Sequence[float],
    *,
    variance_gain: float = 1.0,
    autocorr_gain: float = 1.0,
) -> CriticalSlowingDown:
    """Оценить critical slowing down по окну (чистая).

    Рост дисперсии И lag-1 автокорреляции — ранний признак смены режима
    (Scheffer/Dakos). Обе компоненты нормируются в [0, 1] и смешиваются
    взвешенно:

        v = variance / (variance + 1)        # 0 при покое, → 1 при разбросе
        a = max(0, lag1_autocorrelation)     # 0 при шуме, → 1 при инерции
        slowing = (g_v·v + g_a·a) / (g_v + g_a)

    Спокойный/константный ряд → slowing ≈ 0; растущая дисперсия и
    автокорреляция → slowing → 1. Для короткого ряда (< 3 точек) — безопасный
    дефолт (slowing=0.0).

    Args:
        series: Окно метрики (например, F или max(scores)).
        variance_gain: Вес дисперсионной компоненты (>= 0).
        autocorr_gain: Вес автокорреляционной компоненты (>= 0).

    Returns:
        CriticalSlowingDown со снимком variance/autocorrelation/slowing.

    Raises:
        ValueError: Если gains < 0.
    """
    if variance_gain < 0.0 or autocorr_gain < 0.0:
        raise ValueError(
            f"gains must be >= 0, got {variance_gain}/{autocorr_gain}"
        )
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 3:
        return CriticalSlowingDown(
            variance=float(np.var(arr)) if arr.size else 0.0,
            autocorrelation=0.0,
            slowing=0.0,
            is_warning=False,
        )
    variance = float(np.var(arr))
    autocorr = _lag1_autocorrelation(arr)
    total_gain = variance_gain + autocorr_gain
    if total_gain <= _EPS:
        return CriticalSlowingDown(
            variance=variance,
            autocorrelation=autocorr,
            slowing=0.0,
            is_warning=False,
        )
    v_norm = variance / (variance + 1.0)
    a_norm = max(0.0, autocorr)
    slowing = _clip01(
        (variance_gain * v_norm + autocorr_gain * a_norm) / total_gain
    )
    return CriticalSlowingDown(
        variance=variance,
        autocorrelation=autocorr,
        slowing=slowing,
        is_warning=False,
    )


def classify_change(
    *,
    core_preserved: bool,
    traceable: bool,
    coherent: bool,
    changed: bool = True,
) -> ChangeAssessment:
    """Классифицировать изменение по трём осям INTENT §4 (чистая).

    Все три оси сохранены → DEVELOPMENT (рост, не баг); любая нарушена →
    DRIFT (pathological). ``changed=False`` → STABLE (нечего классифицировать).

    Args:
        core_preserved: Ядро сохранено.
        traceable: Изменение объяснимо опытом.
        coherent: Траектория связная.
        changed: Есть ли изменение вообще.

    Returns:
        ChangeAssessment.
    """
    if not changed:
        return ChangeAssessment(
            kind=ChangeKind.STABLE,
            core_preserved=core_preserved,
            traceable=traceable,
            coherent=coherent,
            reason="no change detected",
        )
    if core_preserved and traceable and coherent:
        return ChangeAssessment(
            kind=ChangeKind.DEVELOPMENT,
            core_preserved=True,
            traceable=True,
            coherent=True,
            reason="core preserved, change traceable to experience, coherent",
        )
    broken = [
        name
        for name, ok in (
            ("core", core_preserved),
            ("traceability", traceable),
            ("coherence", coherent),
        )
        if not ok
    ]
    return ChangeAssessment(
        kind=ChangeKind.DRIFT,
        core_preserved=core_preserved,
        traceable=traceable,
        coherent=coherent,
        reason=f"drift: violated {', '.join(broken)}",
    )


def plan_reset(
    *,
    slowing: CriticalSlowingDown,
    assessment: ChangeAssessment,
    soft_threshold: float = 0.5,
    hard_core_broken: bool = False,
) -> ResetPlan:
    """Спланировать сброс по признаку смены режима и классификации (чистая).

    Порядок решений (ADR-0009 §2):
    - core нарушен → HARD (аварийная остановка; это не «сброс к норме»);
    - DRIFT + warning → FREEZE (regime shift: громкий управляемый переход);
    - warning (без DRIFT) → SOFT (adaptive resetting для поиска);
    - иначе → не триггерить.

    Core не сбрасывается: reset меняет установки/приоритеты, не якоря
    (INTENT §4).

    Args:
        slowing: Признак critical slowing down.
        assessment: Классификация изменения.
        soft_threshold: Порог slowing для SOFT-сброса, [0, 1].
        hard_core_broken: Нарушено ли ядро (аварийный случай).

    Returns:
        ResetPlan.

    Raises:
        ValueError: Если ``soft_threshold`` вне [0, 1].
    """
    if not 0.0 <= soft_threshold <= 1.0:
        raise ValueError(f"soft_threshold must be in [0, 1], got {soft_threshold}")
    if hard_core_broken:
        return ResetPlan(
            level=ResetLevel.HARD,
            reason="core broken: catastrophic drift, emergency stop",
            triggered=True,
        )
    if assessment.kind is ChangeKind.DRIFT and slowing.is_warning:
        return ResetPlan(
            level=ResetLevel.FREEZE,
            reason=f"drift with regime-shift warning ({assessment.reason})",
            triggered=True,
        )
    if slowing.is_warning or slowing.slowing >= soft_threshold:
        return ResetPlan(
            level=ResetLevel.SOFT,
            reason="critical slowing down: adaptive reset for search",
            triggered=True,
        )
    return ResetPlan(
        level=ResetLevel.SOFT,
        reason="no warning: stable",
        triggered=False,
    )
