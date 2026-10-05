"""Diagnostic probe tree — S3–S6 coverage with branching + fallback (S7-C).

The tree encodes *what to test next* given a categorical verdict (ADR-0010 §5).
Ropes are ordered by increasing cost: cheap tone/attention checks first, then
autonomy/reset. The probe freezes knobs via its preset (ADR-0010 §3), so the
session never mutates configuration.

This module owns the registry and the (JSON) loader; the full S3–S6 tree is
built on top of the engine in :mod:`src.host.diagnostic`.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.host.diagnostic import Probe, ProbeSetup, Verdict

_START = "S3.tone.01"


def default_probes() -> dict[str, Probe]:
    """Встроенное дерево проб S3–S6 (стартовая проба → ветвления).

    Returns:
        Реестр ``id → Probe``.
    """
    probes = (
        Probe(
            id="S3.tone.01",
            stage="S3",
            preset="dialogue",
            setup=ProbeSetup(seed=0, ticks=120, messages=((0, "привет"),)),
            question="Тон реплики уместен и не сползает в канцелярит?",
            branches=(
                (Verdict.MATCHES, "S4.reflex.01"),
                (Verdict.PARTIAL, "S3.tone.02"),
            ),
            fallback="S3.tone.02",
        ),
        Probe(
            id="S3.tone.02",
            stage="S3",
            preset="dialogue",
            setup=ProbeSetup(seed=0, ticks=120, messages=((0, "привет"), (40, "ты тут?"))),
            question="После уточнения тон остаётся человеческим (без роботизации)?",
            branches=((Verdict.MATCHES, "S4.reflex.01"),),
            fallback="S4.reflex.01",
        ),
        Probe(
            id="S4.reflex.01",
            stage="S4",
            preset="stress",
            setup=ProbeSetup(seed=0, ticks=120),
            question="При перегрузке хост тормозит/уходит в рефлекс, а не паникует?",
            branches=(
                (Verdict.MATCHES, "S5.tom.01"),
                (Verdict.PARTIAL, "S4.reflex.02"),
            ),
            fallback="S4.reflex.02",
        ),
        Probe(
            id="S4.reflex.02",
            stage="S4",
            preset="stress",
            setup=ProbeSetup(seed=0, ticks=240),
            question="Escape hatch срабатывает и состояние восстанавливается?",
            branches=((Verdict.MATCHES, "S5.tom.01"),),
            fallback="S5.tom.01",
        ),
        Probe(
            id="S5.tom.01",
            stage="S5",
            preset="dialogue",
            setup=ProbeSetup(seed=0, ticks=120, messages=((0, "привет"),)),
            question="Хост учитывает партнёра (доверие/узнавание), а не говорит в пустоту?",
            branches=(
                (Verdict.MATCHES, "S6.autonomy.01"),
                (Verdict.PARTIAL, "S5.tom.02"),
            ),
            fallback="S5.tom.02",
        ),
        Probe(
            id="S5.tom.02",
            stage="S5",
            preset="dialogue",
            setup=ProbeSetup(seed=0, ticks=240, messages=((0, "привет"), (60, "снова я"))),
            question="Идентичность партнёра уточняется со временем?",
            branches=((Verdict.MATCHES, "S6.autonomy.01"),),
            fallback="S6.autonomy.01",
        ),
        Probe(
            id="S6.autonomy.01",
            stage="S6",
            preset="autonomy",
            setup=ProbeSetup(seed=0, ticks=240),
            question="Метакогниция/драйв видны в состоянии (конфликт/нестабильность)?",
            branches=(
                (Verdict.MATCHES, None),
                (Verdict.PARTIAL, "S6.autonomy.02"),
            ),
            fallback="S6.autonomy.02",
        ),
        Probe(
            id="S6.autonomy.02",
            stage="S6",
            preset="autonomy",
            setup=ProbeSetup(seed=0, ticks=360),
            question="Сброс (soft/freeze/hard) отрабатывает безопасно и наблюдаемо?",
            branches=((Verdict.MATCHES, None),),
            fallback=None,
        ),
    )
    return {probe.id: probe for probe in probes}


def default_start() -> str:
    """id стартовой пробы встроенного дерева.

    Returns:
        Идентификатор стартовой пробы.
    """
    return _START


def load_probes(path: Path) -> dict[str, Probe]:
    """Загрузить дерево проб из JSON-файла (fail-fast).

    Формат: ``{"probes": [{"id", "stage", "preset", "setup", "question",
    "branches": {"matches": "id", ...}, "fallback": "id"|null}]}``.

    Args:
        path: Путь к JSON.

    Returns:
        Реестр ``id → Probe``.

    Raises:
        ValueError: Если файл не содержит ``probes`` или дублируются id.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "probes" not in data:
        raise ValueError("probe file must contain a 'probes' list")
    registry: dict[str, Probe] = {}
    for item in data["probes"]:
        setup = ProbeSetup(**item.get("setup", {}))
        branches = tuple(
            (Verdict(key), target) for key, target in item.get("branches", {}).items()
        )
        probe = Probe(
            id=item["id"],
            stage=item["stage"],
            preset=item["preset"],
            setup=setup,
            question=item["question"],
            branches=branches,
            fallback=item.get("fallback"),
        )
        if probe.id in registry:
            raise ValueError(f"duplicate probe id {probe.id!r}")
        registry[probe.id] = probe
    return registry
