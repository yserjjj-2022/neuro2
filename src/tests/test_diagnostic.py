"""Tests for the diagnostic engine Core + Shell (S7-C)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.host.control import ControlChannel
from src.host.diagnostic import (
    DiagnosticSession,
    DiagnosticSnapshot,
    Probe,
    ProbeResult,
    ProbeSetup,
    Verdict,
    journal_records,
    next_probe,
    parse_verdict,
    take_snapshot,
)
from src.host.probes import default_probes, default_start, load_probes


class TestParseVerdict:
    """Разбор категориального ответа."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("matches", Verdict.MATCHES),
            ("MATCHES", Verdict.MATCHES),
            ("1", Verdict.MATCHES),
            ("y", Verdict.MATCHES),
            ("partial", Verdict.PARTIAL),
            ("2", Verdict.PARTIAL),
            ("mismatch", Verdict.MISMATCH),
            ("n", Verdict.MISMATCH),
            ("3", Verdict.MISMATCH),
            ("  match ", Verdict.MATCHES),
        ],
    )
    def test_aliases(self, text: str, expected: Verdict) -> None:
        assert parse_verdict(text) is expected

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_verdict("maybe")


class TestNextProbe:
    """Ветвление по категории + fallback."""

    def test_branch_match(self) -> None:
        probe = Probe(
            "a", "S3", "baseline", ProbeSetup(), "q?",
            branches=((Verdict.MATCHES, "b"),),
        )
        assert next_probe(probe, Verdict.MATCHES) == "b"

    def test_fallback(self) -> None:
        probe = Probe(
            "a", "S3", "baseline", ProbeSetup(), "q?", fallback="c"
        )
        assert next_probe(probe, Verdict.MISMATCH) == "c"

    def test_no_branch_no_fallback(self) -> None:
        probe = Probe("a", "S3", "baseline", ProbeSetup(), "q?")
        assert next_probe(probe, Verdict.PARTIAL) is None


class TestValidation:
    """Валидация Core-объектов."""

    def test_setup_ticks(self) -> None:
        with pytest.raises(ValueError):
            ProbeSetup(ticks=0)

    def test_probe_empty_id(self) -> None:
        with pytest.raises(ValueError):
            Probe("", "S3", "baseline", ProbeSetup(), "q?")

    def test_probe_empty_question(self) -> None:
        with pytest.raises(ValueError):
            Probe("a", "S3", "baseline", ProbeSetup(), "")

    def test_snapshot_frozen(self) -> None:
        import dataclasses

        snap = DiagnosticSnapshot(1, 0.0, 0.0, 0.0, 0.0, "none", 0.0, 0.0, 0.0, "", "")
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.f = 1.0  # type: ignore[misc]


def _session(tmp_path: Path, inputs: list[str], probes: dict) -> DiagnosticSession:
    it = iter(inputs)
    return DiagnosticSession(
        probes=probes,
        input_fn=lambda _prompt: next(it),
        output_fn=lambda _line: None,
        journal_path=tmp_path / "journal.jsonl",
        workdir=tmp_path / "work",
    )


class TestDiagnosticSession:
    """Скрипт → снимок → вердикт → ветвление → журнал."""

    def test_chain_and_journal(self, tmp_path: Path) -> None:
        probes = {
            "a": Probe(
                "a", "S3", "baseline", ProbeSetup(ticks=4), "Q a?",
                branches=((Verdict.MATCHES, "b"),), fallback=None,
            ),
            "b": Probe("b", "S4", "baseline", ProbeSetup(ticks=4), "Q b?"),
        }
        session = _session(tmp_path, ["matches", "good", "mismatch", "bad"], probes)
        results = session.run(start="a")
        assert [r.probe_id for r in results] == ["a", "b"]
        assert results[0].verdict is Verdict.MATCHES
        assert results[0].next_probe == "b"
        assert results[1].next_probe is None
        records = journal_records(tmp_path / "journal.jsonl")
        assert len(records) == 2
        assert records[0]["verdict"] == "matches"
        assert records[0]["comment"] == "good"
        assert records[0]["snapshot"]["tick"] >= 1

    def test_snapshot_attached(self, tmp_path: Path) -> None:
        probes = {"a": Probe("a", "S3", "baseline", ProbeSetup(ticks=3), "Q?")}
        session = _session(tmp_path, ["partial", ""], probes)
        result = session.run_probe(probes["a"])
        assert isinstance(result, ProbeResult)
        assert isinstance(result.snapshot, DiagnosticSnapshot)
        assert result.snapshot.task != ""

    def test_max_probes(self, tmp_path: Path) -> None:
        probes = {
            "a": Probe(
                "a", "S3", "baseline", ProbeSetup(ticks=2), "Q?",
                branches=((Verdict.MATCHES, "b"),),
            ),
            "b": Probe(
                "b", "S3", "baseline", ProbeSetup(ticks=2), "Q?",
                branches=((Verdict.MATCHES, "c"),),
            ),
            "c": Probe("c", "S3", "baseline", ProbeSetup(ticks=2), "Q?"),
        }
        session = _session(tmp_path, ["matches", ""] * 3, probes)
        results = session.run(start="a", max_probes=2)
        assert [r.probe_id for r in results] == ["a", "b"]

    def test_unknown_start_raises(self, tmp_path: Path) -> None:
        session = _session(tmp_path, [], {"a": Probe("a", "S3", "baseline", ProbeSetup(ticks=2), "Q?")})
        with pytest.raises(ValueError):
            session.run(start="nope")

    def test_branch_to_unknown_raises(self, tmp_path: Path) -> None:
        probes = {
            "a": Probe(
                "a", "S3", "baseline", ProbeSetup(ticks=2), "Q?",
                branches=((Verdict.MATCHES, "ghost"),),
            )
        }
        session = _session(tmp_path, ["matches", ""], probes)
        with pytest.raises(ValueError):
            session.run(start="a")

    def test_no_journal_when_none(self, tmp_path: Path) -> None:
        probes = {"a": Probe("a", "S3", "baseline", ProbeSetup(ticks=2), "Q?")}
        it = iter(["matches", ""])
        session = DiagnosticSession(
            probes=probes,
            input_fn=lambda _p: next(it),
            output_fn=lambda _l: None,
            workdir=tmp_path / "w",
        )
        session.run(start="a")
        assert not (tmp_path / "journal.jsonl").exists()


class TestSnapshot:
    """take_snapshot + ControlChannel.snapshot."""

    def test_take_snapshot_after_run(self, tmp_path: Path) -> None:
        from src.config import load_preset
        from src.host.loop import build_host_loop
        from src.host.sensitivity import DeterministicMeter

        config = load_preset("baseline")
        loop = build_host_loop(config, meter=DeterministicMeter())
        loop.run(5)
        try:
            snap = take_snapshot(loop)
            channel = ControlChannel(loop)
            assert snap == channel.snapshot()
            assert snap.tick >= 1
        finally:
            loop.close()


class TestProbes:
    """Встроенное дерево + загрузчик."""

    def test_default_tree_valid(self) -> None:
        probes = default_probes()
        assert default_start() in probes
        for probe in probes.values():
            for _verdict, target in probe.branches:
                assert target is None or target in probes
            assert probe.fallback is None or probe.fallback in probes

    def test_default_tree_reachable(self) -> None:
        probes = default_probes()
        seen: set[str] = set()
        current: str | None = default_start()
        while current is not None:
            assert current in probes
            assert current not in seen
            seen.add(current)
            probe = probes[current]
            current = probe.branches[0][1] if probe.branches else probe.fallback

    def test_load_probes_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "probes.json"
        path.write_text(
            '{"probes": [{"id": "x", "stage": "S3", "preset": "baseline", '
            '"setup": {"ticks": 5}, "question": "Q?", '
            '"branches": {"matches": "x"}, "fallback": null}]}'
        )
        probes = load_probes(path)
        assert probes["x"].setup.ticks == 5
        assert probes["x"].branches == ((Verdict.MATCHES, "x"),)

    def test_load_probes_bad(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{}")
        with pytest.raises(ValueError):
            load_probes(path)
