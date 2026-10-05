"""Validity tests for the S3–S6 diagnostic probe tree (S7-C)."""

from __future__ import annotations

import pytest

from src.config import available_presets
from src.host.diagnostic import Probe, ProbeSetup, Verdict, next_probe
from src.host.probes import default_probes, default_start

_STAGES = {"S3", "S4", "S5", "S6"}


@pytest.fixture(scope="module")
def probes() -> dict[str, Probe]:
    return default_probes()


class TestTreeValidity:
    """Структурные инварианты дерева проб."""

    def test_start_present(self, probes: dict[str, Probe]) -> None:
        assert default_start() in probes

    def test_unique_ids(self, probes: dict[str, Probe]) -> None:
        assert len(probes) == len(set(probes))

    def test_keys_match_ids(self, probes: dict[str, Probe]) -> None:
        for key, probe in probes.items():
            assert key == probe.id

    def test_all_branch_targets_exist(self, probes: dict[str, Probe]) -> None:
        for probe in probes.values():
            for _verdict, target in probe.branches:
                assert target is None or target in probes, (probe.id, target)

    def test_fallback_targets_exist(self, probes: dict[str, Probe]) -> None:
        for probe in probes.values():
            assert probe.fallback is None or probe.fallback in probes

    def test_no_duplicate_branch_verdicts(self, probes: dict[str, Probe]) -> None:
        for probe in probes.values():
            verdicts = [v for v, _ in probe.branches]
            assert len(verdicts) == len(set(verdicts)), probe.id

    def test_stage_coverage(self, probes: dict[str, Probe]) -> None:
        assert {p.stage for p in probes.values()} == _STAGES

    def test_presets_are_known(self, probes: dict[str, Probe]) -> None:
        known = set(available_presets())
        for probe in probes.values():
            assert probe.preset in known, probe.id

    def test_setups_valid(self, probes: dict[str, Probe]) -> None:
        for probe in probes.values():
            assert isinstance(probe.setup, ProbeSetup)
            assert probe.setup.ticks >= 1
            for tick, text in probe.setup.messages:
                assert tick >= 0
                assert text


class TestReachability:
    """Дерево связно и любая ветка завершается (без циклов)."""

    def test_terminates_on_first_branch(self, probes: dict[str, Probe]) -> None:
        seen: set[str] = set()
        current: str | None = default_start()
        while current is not None:
            assert current in probes
            assert current not in seen, f"cycle at {current}"
            seen.add(current)
            probe = probes[current]
            current = (
                probe.branches[0][1]
                if probe.branches
                else probe.fallback
            )

    def test_all_probes_reachable_from_start(
        self, probes: dict[str, Probe]
    ) -> None:
        # BFS от старта по всем веткам/fallback: нет «мёртвых» проб.
        seen: set[str] = set()
        stack: list[str | None] = [default_start()]
        while stack:
            pid = stack.pop()
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            probe = probes[pid]
            stack.extend(target for _verdict, target in probe.branches)
            stack.append(probe.fallback)
        assert seen == set(probes)

    def test_every_nonterminal_has_way_forward(self, probes: dict[str, Probe]) -> None:
        for probe in probes.values():
            has_forward = any(t is not None for _v, t in probe.branches) or (
                probe.fallback is not None
            )
            # Терминальные пробы (S6) могут не иметь ветвлений.
            if probe.stage != "S6":
                assert has_forward, probe.id


class TestBranchSemantics:
    """next_probe ведёт себя согласованно с деревом."""

    def test_matches_advances(self, probes: dict[str, Probe]) -> None:
        probe = probes[default_start()]
        target = next_probe(probe, Verdict.MATCHES)
        assert target in probes

    def test_fallback_on_unlisted_verdict(self, probes: dict[str, Probe]) -> None:
        probe = Probe(
            "x", "S3", "baseline", ProbeSetup(ticks=2), "q?", fallback="y"
        )
        assert next_probe(probe, Verdict.MISMATCH) == "y"
