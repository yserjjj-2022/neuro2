"""Diagnostic session — scripted probes with categorical verdicts (S7-C).

Functional Core (pure, ADR-0010 §3–§5):

* :class:`Verdict` — ordinal category (``matches``/``partial``/``mismatch``).
* :class:`ProbeSetup` / :class:`Probe` — a probe: id, stage, preset, scripted
  run and the observer's question + branching rule.
* :func:`next_probe` — pick the next probe id from the categorical verdict.
* :class:`DiagnosticSnapshot` — the numeric state attached to a verdict (never
  judged by the human).
* :class:`ProbeResult` — verdict + comment + snapshot + next probe.

Imperative Shell:

* :class:`DiagnosticSession` — runs a probe (scripted loop), takes a snapshot,
  asks the observer, records the verdict and follows the branch. The knobs are
  **frozen by the preset**: the session never mutates config (ADR-0010 §3).
  I/O is injected (``input_fn``/``output_fn``), the journal is a JSONL file.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from src.config import load_preset
from src.host.loop import HostLoop, build_host_loop
from src.host.sensitivity import DeterministicMeter


class Verdict(Enum):
    """Категориальный вердикт наблюдателя (порядковая шкала)."""

    MATCHES = "matches"
    PARTIAL = "partial"
    MISMATCH = "mismatch"


_VERDICT_ALIASES: dict[str, Verdict] = {
    "matches": Verdict.MATCHES,
    "match": Verdict.MATCHES,
    "m": Verdict.MATCHES,
    "y": Verdict.MATCHES,
    "1": Verdict.MATCHES,
    "partial": Verdict.PARTIAL,
    "part": Verdict.PARTIAL,
    "p": Verdict.PARTIAL,
    "2": Verdict.PARTIAL,
    "mismatch": Verdict.MISMATCH,
    "no": Verdict.MISMATCH,
    "n": Verdict.MISMATCH,
    "x": Verdict.MISMATCH,
    "3": Verdict.MISMATCH,
}


def parse_verdict(text: str) -> Verdict:
    """Разобрать ответ наблюдателя в категорию (чистая).

    Args:
        text: Ввод (``matches``/``partial``/``mismatch`` или 1/2/3, y/n).

    Returns:
        Verdict.

    Raises:
        ValueError: Если ответ не распознан.
    """
    key = text.strip().lower()
    if key not in _VERDICT_ALIASES:
        raise ValueError(f"unrecognized verdict {text!r}")
    return _VERDICT_ALIASES[key]


@dataclass(frozen=True)
class ProbeSetup:
    """Скрипт прогона пробы (детерминированный).

    Attributes:
        seed: Зерно провайдеров.
        ticks: Число тиков прогона.
        messages: Скрипт коммуникативных сообщений ``(tick, text)``.
    """

    seed: int = 0
    ticks: int = 120
    messages: tuple[tuple[int, str], ...] = ()

    def __post_init__(self) -> None:
        if self.ticks < 1:
            raise ValueError(f"ticks must be >= 1, got {self.ticks}")


@dataclass(frozen=True)
class Probe:
    """Одна диагностическая проба (ADR-0010 §5).

    Attributes:
        id: Идентификатор (например ``S3.tone.01``).
        stage: Покрываемая стадия (S3/S4/S5/S6/C10).
        preset: Имя пресета (ручки заморожены).
        setup: Скрипт прогона (seed/ticks/messages).
        question: Вопрос наблюдателю.
        branches: Правило ветвления ``verdict → id`` следующей пробы
            (``None`` — конец ветки).
        fallback: Проба по умолчанию, если ветвление не задано.
    """

    id: str
    stage: str
    preset: str
    setup: ProbeSetup
    question: str
    branches: tuple[tuple[Verdict, str | None], ...] = ()
    fallback: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("probe id must not be empty")
        if not self.question:
            raise ValueError("probe question must not be empty")


def next_probe(probe: Probe, verdict: Verdict) -> str | None:
    """Выбрать следующую пробу по категориальному ответу (чистая).

    Args:
        probe: Текущая проба.
        verdict: Вердикт наблюдателя.

    Returns:
        id следующей пробы или None (конец ветки).
    """
    for branch_verdict, target in probe.branches:
        if branch_verdict is verdict:
            return target
    return probe.fallback


@dataclass(frozen=True)
class DiagnosticSnapshot:
    """Числовой снимок состояния на момент вердикта (не оценивается человеком).

    Attributes:
        tick: Номер тика.
        f: Свободная энергия.
        valence: Валентность.
        stress: Аллостатический стресс.
        gamma: Точность γ.
        task: Активная задача/аттрактор.
        partner_trust: Доверие к партнёру (S5).
        partner_uncertainty: Неопределённость идентичности (S5).
        metacog_conflict: Несогласие ансамбля колонок (S6).
        reset_level: Уровень сброса ("", soft/freeze/hard) — S6.
        change_kind: Классификация изменения (S6).
    """

    tick: int
    f: float
    valence: float
    stress: float
    gamma: float
    task: str
    partner_trust: float
    partner_uncertainty: float
    metacog_conflict: float
    reset_level: str
    change_kind: str


@dataclass(frozen=True)
class ProbeResult:
    """Итог пробы: категория + комментарий + числовой снимок + ветвление.

    Attributes:
        probe_id: Идентификатор пробы.
        verdict: Категориальный вердикт.
        comment: Свободный комментарий наблюдателя.
        snapshot: Числовой снимок состояния.
        next_probe: id следующей пробы (None → конец).
    """

    probe_id: str
    verdict: Verdict
    comment: str
    snapshot: DiagnosticSnapshot
    next_probe: str | None


LoopFactory = Callable[[str, ProbeSetup, Path], HostLoop]


def take_snapshot(loop: HostLoop) -> DiagnosticSnapshot:
    """Снять числовой снимок состояния с loop (чистая, S7-C).

    Единый источник снимка для ``DiagnosticSession`` и ``ControlChannel``.

    Args:
        loop: Host loop.

    Returns:
        DiagnosticSnapshot.
    """
    outcome = loop.last_outcome
    return DiagnosticSnapshot(
        tick=loop.current_tick,
        f=outcome.result.f if outcome is not None else 0.0,
        valence=outcome.result.valence if outcome is not None else 0.0,
        stress=outcome.result.allostatic_stress if outcome is not None else 0.0,
        gamma=outcome.result.gamma if outcome is not None else 0.0,
        task=loop.active_task(),
        partner_trust=loop.last_partner_trust,
        partner_uncertainty=loop.last_partner_uncertainty,
        metacog_conflict=loop.last_metacog_conflict,
        reset_level=loop.last_reset_level,
        change_kind=loop.last_change_kind,
    )


_EMPTY_SNAPSHOT = DiagnosticSnapshot(
    0, 0.0, 0.0, 0.0, 0.0, "none", 0.0, 0.0, 0.0, "", ""
)


def _build_probe_loop(preset: str, setup: ProbeSetup, workdir: Path) -> HostLoop:
    """Собрать детерминированный loop под пробу (пресет замораживает ручки)."""
    workdir.mkdir(parents=True, exist_ok=True)
    config = load_preset(preset)
    config = replace(
        config,
        seed=setup.seed,
        max_ticks=setup.ticks,
        log_path=str(workdir / "run.jsonl"),
        memory=replace(config.memory, db_path=str(workdir / "mem.db")),
    )
    return build_host_loop(
        config, meter=DeterministicMeter(), messages=setup.messages
    )


@dataclass
class DiagnosticSession:
    """Shell: проба → снимок → вопрос → вердикт → ветвление → журнал.

    Attributes:
        probes: Реестр проб ``id → Probe``.
        input_fn: Источник ответов наблюдателя (инъекция).
        output_fn: Приёмник вопросов/статуса (инъекция).
        journal_path: Путь к JSONL-журналу вердиктов (None → без записи).
        workdir: Каталог для логов/БД прогонов.
        loop_factory: Сборка loop под пробу (инъекция для тестов).
    """

    probes: Mapping[str, Probe]
    input_fn: Callable[[str], str] = input
    output_fn: Callable[[str], None] = print
    journal_path: Path | None = None
    workdir: Path = field(default_factory=lambda: Path(".diagnostic"))
    loop_factory: LoopFactory = _build_probe_loop
    _loop: HostLoop | None = field(default=None, init=False, repr=False)

    def snapshot(self) -> DiagnosticSnapshot:
        """Числовой снимок текущего состояния loop (не оценивается).

        Returns:
            DiagnosticSnapshot; нулевой, если loop ещё не запущен.
        """
        loop = self._loop
        if loop is None:
            return _EMPTY_SNAPSHOT
        return take_snapshot(loop)

    def run_probe(self, probe: Probe) -> ProbeResult:
        """Прогнать одну пробу: скрипт → снимок → вопрос → вердикт → журнал.

        Args:
            probe: Проба.

        Returns:
            ProbeResult.
        """
        self._loop = self.loop_factory(
            probe.preset, probe.setup, self.workdir / probe.id
        )
        try:
            self._loop.run(probe.setup.ticks)
            snapshot = self.snapshot()
        finally:
            self._loop.close()
            self._loop = None

        self.output_fn(f"[{probe.id}] {probe.question}")
        verdict = parse_verdict(self.input_fn("verdict [matches/partial/mismatch]: "))
        comment = self.input_fn("comment (optional): ")
        result = ProbeResult(
            probe_id=probe.id,
            verdict=verdict,
            comment=comment,
            snapshot=snapshot,
            next_probe=next_probe(probe, verdict),
        )
        self._journal(result)
        return result

    def run(self, *, start: str, max_probes: int = 0) -> list[ProbeResult]:
        """Прогнать цепочку проб от ``start`` по ветвлениям.

        Args:
            start: id стартовой пробы.
            max_probes: Максимум проб (0 → до конца цепочки).

        Returns:
            Список ProbeResult в порядке прохождения.

        Raises:
            ValueError: Если стартовая проба отсутствует или ветвление ведёт
                на неизвестную пробу.
        """
        if start not in self.probes:
            raise ValueError(f"unknown start probe {start!r}")
        results: list[ProbeResult] = []
        current: str | None = start
        while current is not None and (max_probes == 0 or len(results) < max_probes):
            if current not in self.probes:
                raise ValueError(f"branch leads to unknown probe {current!r}")
            result = self.run_probe(self.probes[current])
            results.append(result)
            current = result.next_probe
        return results

    def _journal(self, result: ProbeResult) -> None:
        """Дописать вердикт в JSONL-журнал (crash-safe append)."""
        if self.journal_path is None:
            return
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "probe_id": result.probe_id,
            "verdict": result.verdict.value,
            "comment": result.comment,
            "snapshot": dataclasses.asdict(result.snapshot),
            "next_probe": result.next_probe,
        }
        with open(self.journal_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def journal_records(path: Path) -> Sequence[dict]:
    """Прочитать журнал диагностики (чистая обёртка над JSONL).

    Args:
        path: Путь к JSONL-журналу.

    Returns:
        Список записей-словарей.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.split("\n")]
