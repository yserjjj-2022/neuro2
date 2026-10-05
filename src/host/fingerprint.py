"""Behavioral fingerprint — compact numeric summary of a run (S7, VALIDATION §5).

Functional Core (pure, side-effect-free): a run's telemetry is reduced to a
small frozen vector so that two runs can be compared (regression) and so that a
sensitivity harness can track a single named metric while a knob is perturbed.

Input is a sequence of telemetry rows — the JSONL objects produced by
``TelemetryLogger`` (``Mapping[str, Any]``). Reading rows as mappings keeps the
fingerprint independent of the telemetry dataclass and matches how replay and
diagnostics consume logs (ADR-0010 §3, §7).

Two entry points:

* :func:`behavioral_fingerprint` — the S7 ``BehavioralFingerprint`` used by the
  sensitivity harness and the diagnostic session.
* :func:`regression_fingerprint` — the richer per-field summary historically
  used by the behavioral regression suite (kept verbatim, re-exported by the
  test helper).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Fields of the telemetry JSONL row that the fingerprint consumes. Absent keys
# are treated as their neutral default (0 / "" / False).
_REQUIRED_KEYS = ("free_energy", "allostatic_stress", "valence", "reflex_tags")


@dataclass(frozen=True)
class FProfile:
    """Сводка свободной энергии F по прогону.

    Attributes:
        mean: Среднее F.
        std: Стандартное отклонение F.
        max: Максимум F.
    """

    mean: float
    std: float
    max: float


@dataclass(frozen=True)
class BehavioralFingerprint:
    """Компактный числовой отпечаток прогона (VALIDATION §5, S7-A).

    Attributes:
        f_profile: Сводка F (mean/std/max).
        reflex_count: Число reflex-событий.
        stress_peaks: Пиковое значение аллостатического стресса.
        active_fraction: Доля тиков хотя бы с одной активной колонкой, [0, 1].
        resource_alarms: Число тиков с ресурсным алярмом (reflex-тег resources).
        talk_rate: Доля тиков с репликой, [0, 1].
        throttle_rate: Доля тиков под throttle, [0, 1].
        explore_rate: Доля тиков с policy-действием EXPLORE, [0, 1].
        initiative_rate: Доля тиков с policy-действием INITIATIVE, [0, 1].
    """

    f_profile: FProfile
    reflex_count: int
    stress_peaks: float
    active_fraction: float
    resource_alarms: int
    talk_rate: float
    throttle_rate: float
    explore_rate: float
    initiative_rate: float

    def metric(self, name: str) -> float:
        """Значение именованной метрики отпечатка (для harness).

        Args:
            name: Имя метрики (``f_mean``, ``talk_rate``, ``explore_rate``, ...).

        Returns:
            Числовое значение метрики.

        Raises:
            KeyError: Если метрика неизвестна.
        """
        table: dict[str, float] = {
            "f_mean": self.f_profile.mean,
            "f_std": self.f_profile.std,
            "f_max": self.f_profile.max,
            "reflex_count": float(self.reflex_count),
            "stress_peaks": self.stress_peaks,
            "active_fraction": self.active_fraction,
            "resource_alarms": float(self.resource_alarms),
            "talk_rate": self.talk_rate,
            "throttle_rate": self.throttle_rate,
            "explore_rate": self.explore_rate,
            "initiative_rate": self.initiative_rate,
        }
        if name not in table:
            raise KeyError(f"unknown fingerprint metric {name!r}")
        return table[name]


# Порядок метрик в расстоянии (детерминизм).
FINGERPRINT_METRICS: tuple[str, ...] = (
    "f_mean",
    "f_std",
    "f_max",
    "reflex_count",
    "stress_peaks",
    "active_fraction",
    "resource_alarms",
    "talk_rate",
    "throttle_rate",
    "explore_rate",
    "initiative_rate",
)


def _column(events: Sequence[Mapping[str, Any]], key: str) -> list[Any]:
    """Извлечь колонку значений из строк телеметрии."""
    return [event[key] for event in events]


def _rate(events: Sequence[Mapping[str, Any]], key: str) -> float:
    """Доля тиков с истинным флагом ``key`` (0.0 при отсутствии ключа)."""
    if not events:
        return 0.0
    return sum(1 for event in events if event.get(key)) / len(events)


def behavioral_fingerprint(
    events: Sequence[Mapping[str, Any]],
) -> BehavioralFingerprint:
    """Собрать отпечаток из телеметрии (чистая функция).

    Args:
        events: Строки телеметрии (JSONL-объекты) в порядке тиков.

    Returns:
        BehavioralFingerprint.

    Raises:
        ValueError: Если ``events`` пуст или в строке нет обязательных полей.
    """
    if not events:
        raise ValueError("events must not be empty")
    for event in events:
        for key in _REQUIRED_KEYS:
            if key not in event:
                raise ValueError(f"telemetry event missing key {key!r}")

    f_values = np.asarray(_column(events, "free_energy"), dtype=np.float64)
    stress_values = np.asarray(_column(events, "allostatic_stress"), dtype=np.float64)

    reflex_count = sum(1 for event in events if event["reflex_tags"])
    resource_alarms = sum(
        1 for event in events if "resources" in str(event["reflex_tags"]).split(",")
    )
    active_fraction = sum(
        1 for event in events if int(event.get("active_columns", 0)) > 0
    ) / len(events)

    return BehavioralFingerprint(
        f_profile=FProfile(
            mean=float(np.mean(f_values)),
            std=float(np.std(f_values)),
            max=float(np.max(f_values)),
        ),
        reflex_count=reflex_count,
        stress_peaks=float(np.max(stress_values)),
        active_fraction=active_fraction,
        resource_alarms=resource_alarms,
        talk_rate=_rate(events, "spoke"),
        throttle_rate=_rate(events, "throttle"),
        explore_rate=sum(
            1 for event in events if event.get("policy_action", "") == "explore"
        )
        / len(events),
        initiative_rate=sum(
            1 for event in events if event.get("policy_action", "") == "initiative"
        )
        / len(events),
    )


def fingerprint_distance(a: BehavioralFingerprint, b: BehavioralFingerprint) -> float:
    """Нормированное расстояние между отпечатками (чистая), >= 0.

    Каждая метрика нормируется в [0, 1) как ``|a-b| / (1 + |a| + |b|)`` и
    усредняется по метрикам: 0.0 при равенстве, симметрично.

    Args:
        a: Первый отпечаток.
        b: Второй отпечаток.

    Returns:
        Среднее нормированное расстояние в [0, 1).
    """
    diffs = []
    for name in FINGERPRINT_METRICS:
        va = a.metric(name)
        vb = b.metric(name)
        diffs.append(abs(va - vb) / (1.0 + abs(va) + abs(vb)))
    return float(sum(diffs) / len(diffs))


def regression_fingerprint(
    events: Sequence[Mapping[str, Any]],
    valence_significance: float = 1.0,
) -> dict[str, float]:
    """Компактная сводка прогона для регресса (VALIDATION §5).

    Историческая (более подробная, чем ``BehavioralFingerprint``) сводка,
    используемая поведенческим регрессом. Перенесена в Core из тестов без
    изменения поведения.

    Чистая функция: не мутирует ``events``.

    Args:
        events: Список событий телеметрии (dict из JSONL).
        valence_significance: Порог |valence|, ниже которого колебания
            считаются микро-шумом у нуля (не «сменой настроения»).

    Returns:
        Словарь метрик: профиль F, стресс, valence, reflex, drift, latency.

    Raises:
        ValueError: Если ``events`` пуст.
    """
    if not events:
        raise ValueError("events must not be empty")

    f_values = [e["free_energy"] for e in events]
    stress_values = [e["allostatic_stress"] for e in events]
    valence_values = [e["valence"] for e in events]
    latency_values = [e["latency_ms"] for e in events]

    # Значимые смены знака — только среди |valence| > порога
    significant = [v for v in valence_values if abs(v) > valence_significance]
    sign_changes = sum(
        1 for a, b in itertools.pairwise(significant) if (a > 0) != (b > 0)
    )
    reflex_events = sum(1 for e in events if e["reflex_tags"])

    return {
        "n_events": float(len(events)),
        "f_min": float(min(f_values)),
        "f_max": float(max(f_values)),
        "f_final": float(f_values[-1]),
        "stress_max": float(max(stress_values)),
        "stress_final": float(stress_values[-1]),
        "valence_sign_changes": float(sign_changes),
        "reflex_events": float(reflex_events),
        "drift_events": float(sum(1 for e in events if e["drift"])),
        "memory_hits": float(sum(1 for e in events if e.get("memory_hit"))),
        "episodes_stored": float(sum(1 for e in events if e.get("episode_stored"))),
        "throttle_events": float(sum(1 for e in events if e.get("throttle"))),
        "policy_events": float(
            sum(1 for e in events if e.get("policy_action", "") != "")
        ),
        "latency_p50_ms": float(np.percentile(latency_values, 50)),
        "latency_p95_ms": float(np.percentile(latency_values, 95)),
    }
