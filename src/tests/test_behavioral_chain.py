"""Tests for the behavioral chain harness Core + Shell (VALIDATION §7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.policy import Action, PolicyCandidate, PolicyTrace
from src.host.behavioral_chain import (
    Ablation,
    AblationCheck,
    BehavioralChainRunner,
    IntentInvariant,
    IntentView,
    ReactionClass,
    Scenario,
    StateInvariant,
    StateView,
    check_intent,
    check_state,
    classify_reaction,
    default_ablations,
    default_scenarios,
    intent_from_state,
    summarize,
)
from src.speech.intent import build_intent_frame


def _trace(action: Action) -> PolicyTrace:
    """Собрать минимальную трассу с заданным действием."""
    return PolicyTrace(
        chosen=action,
        reason=f"test {action.value}",
        candidates=(
            PolicyCandidate(action, pragmatic=1.0, epistemic=0.0, value=1.0, reason="t"),
        ),
    )


def _view(**overrides: object) -> StateView:
    """Снимок состояния с безопасными дефолтами и точечными переопределениями."""
    base: dict[str, object] = {
        "tick": 0,
        "f": 1.0,
        "valence": 0.0,
        "stress": 0.0,
        "gamma": 1.0,
        "task": "none",
        "active_columns": 1,
        "drift": False,
        "partner_trust": 0.0,
        "partner_uncertainty": 0.0,
    }
    base.update(overrides)
    return StateView(**base)  # type: ignore[arg-type]


class TestClassifyReaction:
    """Класс реакции: решение → наблюдаемый класс (Core)."""

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            (Action.RESPOND, ReactionClass.RESPOND),
            (Action.SILENT, ReactionClass.SILENT),
            (Action.INITIATIVE, ReactionClass.INITIATIVE),
            (Action.IDENTIFY_PARTNER, ReactionClass.IDENTIFY_PARTNER),
            (Action.EXPLORE, ReactionClass.EXPLORE),
        ],
    )
    def test_action_mapping(self, action: Action, expected: ReactionClass) -> None:
        assert classify_reaction(_trace(action)) is expected

    def test_escape_hatch_overrides_policy(self) -> None:
        """Escape hatch приоритетнее решения policy (право голоса)."""
        assert (
            classify_reaction(_trace(Action.SILENT), escape_hatch=True)
            is ReactionClass.ESCAPE_HATCH
        )


class TestCheckState:
    """Инварианты состояния: чувствительность к нарушениям (Core)."""

    def test_clean_passes(self) -> None:
        assert check_state(_view(), tuple(StateInvariant)) == ()

    def test_nan_fails_finite(self) -> None:
        assert "finite" in check_state(
            _view(f=float("nan")), (StateInvariant.FINITE,)
        )

    def test_inf_fails_finite(self) -> None:
        assert "finite" in check_state(
            _view(gamma=float("inf")), (StateInvariant.FINITE,)
        )

    def test_negative_f_fails(self) -> None:
        assert "f_nonneg" in check_state(
            _view(f=-0.1), (StateInvariant.F_NONNEG,)
        )

    def test_negative_stress_fails(self) -> None:
        assert "stress_nonneg" in check_state(
            _view(stress=-0.1), (StateInvariant.STRESS_NONNEG,)
        )

    def test_partner_out_of_range_fails(self) -> None:
        assert "partner_bounded" in check_state(
            _view(partner_trust=1.5), (StateInvariant.PARTNER_BOUNDED,)
        )

    def test_only_requested_invariants_checked(self) -> None:
        """Непроверяемый инвариант не всплывает."""
        assert check_state(_view(f=-1.0), (StateInvariant.FINITE,)) == ()


def _intent_view(**overrides: object) -> IntentView:
    """Снимок интента с grounding-дефолтами и точечными переопределениями."""
    view = intent_from_state(_view(), Action.RESPOND)
    if not overrides:
        return view
    frame = build_intent_frame(
        f=view.frame.free_energy,
        valence=view.frame.valence,
        stress=view.frame.stress,
        task=view.frame.task,
        goal=view.frame.goal,
    )
    base: dict[str, object] = {
        "frame": frame,
        "action": view.action,
        "state_valence": view.state_valence,
        "state_stress": view.state_stress,
        "state_task": view.state_task,
    }
    base.update(overrides)
    return IntentView(**base)  # type: ignore[arg-type]


class TestIntentFromState:
    """Сборка интента из состояния и решения (Core, звено 3)."""

    def test_goal_from_action(self) -> None:
        assert intent_from_state(_view(), Action.EXPLORE).frame.goal == "explore"

    def test_none_action_falls_back_to_respond(self) -> None:
        assert intent_from_state(_view(), None).frame.goal == "respond"

    def test_state_echoed_verbatim(self) -> None:
        view = _view(f=3.0, valence=-2.0, stress=4.0, task="tone")
        intent = intent_from_state(view, Action.RESPOND)
        assert intent.frame.free_energy == 3.0
        assert intent.frame.valence == -2.0
        assert intent.frame.stress == 4.0
        assert intent.frame.task == "tone"
        assert intent.frame.affect == "напряжённое, негативное"


class TestCheckIntent:
    """Grounding интента: соответствие состояния/решения (Core, звено 3)."""

    def test_clean_passes(self) -> None:
        assert check_intent(_intent_view(), tuple(IntentInvariant)) == ()

    def test_goal_ungrounded_fails(self) -> None:
        """Цель не из GOAL_INSTRUCTIONS → нарушение grounding."""
        bad = build_intent_frame(
            f=1.0, valence=0.0, stress=0.0, task="none", goal="bogus"
        )
        assert "goal_consistent" in check_intent(
            _intent_view(frame=bad), tuple(IntentInvariant)
        )

    def test_goal_action_mismatch_fails(self) -> None:
        """Frame говорит respond, а policy выбрала EXPLORE → рассогласование."""
        frame = build_intent_frame(
            f=1.0, valence=0.0, stress=0.0, task="none", goal="respond"
        )
        assert "goal_consistent" in check_intent(
            _intent_view(frame=frame, action=Action.EXPLORE), tuple(IntentInvariant)
        )

    def test_affect_mismatch_fails(self) -> None:
        """Числа frame'а не выводятся из состояния → нарушение grounding."""
        assert "affect_consistent" in check_intent(
            _intent_view(state_valence=-5.0), tuple(IntentInvariant)
        )

    def test_task_mismatch_fails(self) -> None:
        assert "task_consistent" in check_intent(
            _intent_view(state_task="другая"), tuple(IntentInvariant)
        )

    def test_nan_valence_fails(self) -> None:
        frame = build_intent_frame(
            f=1.0, valence=float("nan"), stress=0.0, task="none"
        )
        assert "affect_consistent" in check_intent(
            _intent_view(
                frame=frame, state_valence=float("nan")
            ),
            tuple(IntentInvariant),
        )


class TestScenario:
    """Валидация ячейки сценария (Core)."""

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError):
            Scenario(id="", link="state", preset="baseline")

    def test_bad_link_raises(self) -> None:
        with pytest.raises(ValueError):
            Scenario(id="x", link="nope", preset="baseline")

    def test_bad_ticks_raises(self) -> None:
        with pytest.raises(ValueError):
            Scenario(id="x", link="state", preset="baseline", ticks=0)

    def test_non_born_precondition_raises(self) -> None:
        """primed/matured ещё не реализованы (VALIDATION §7.8)."""
        with pytest.raises(ValueError):
            Scenario(id="x", link="state", preset="baseline", precondition="primed")


class TestIntentRunner:
    """Shell: звено 3 (интент) прогоняется и не даёт нарушений grounding."""

    def test_intent_scenarios_pass(self, tmp_path: Path) -> None:
        runner = BehavioralChainRunner(workdir=tmp_path)
        results = [
            r for r in runner.run_all() if r.scenario.link == "intent"
        ]
        assert results
        assert all(r.passed for r in results)
        assert all(r.intent_violations == () for r in results)


class TestDefaultScenarios:
    """Встроенный корпус звеньев 1–3."""

    def test_covers_state_decisions_intent(self) -> None:
        links = {s.link for s in default_scenarios()}
        assert links == {"state", "decision", "intent"}

    def test_ids_unique(self) -> None:
        ids = [s.id for s in default_scenarios()]
        assert len(ids) == len(set(ids))


class TestAblationCheck:
    """Валидация проверки атрибуции (Core)."""

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError):
            AblationCheck(
                id="", preset="dialogue", mechanism=Ablation.POLICY, observable="f_mean"
            )

    def test_bad_ticks_raises(self) -> None:
        with pytest.raises(ValueError):
            AblationCheck(
                id="x",
                preset="dialogue",
                mechanism=Ablation.POLICY,
                observable="f_mean",
                ticks=0,
            )

    def test_unknown_observable_raises(self) -> None:
        with pytest.raises(ValueError):
            AblationCheck(
                id="x",
                preset="dialogue",
                mechanism=Ablation.POLICY,
                observable="nope",
            )

    def test_as_scenario_carries_preset_and_messages(self) -> None:
        check = AblationCheck(
            id="x",
            preset="dialogue",
            mechanism=Ablation.RECALL,
            observable="f_mean",
            messages=((0, "привет"),),
        )
        scenario = check.as_scenario()
        assert scenario.preset == "dialogue"
        assert scenario.messages == ((0, "привет"),)
        assert scenario.link == "state"


class TestDefaultAblations:
    """Встроенный корпус проверок атрибуции."""

    def test_covers_all_mechanisms(self) -> None:
        mechanisms = {c.mechanism for c in default_ablations()}
        assert mechanisms == {Ablation.POLICY, Ablation.MEMORY, Ablation.RECALL}

    def test_ids_unique(self) -> None:
        ids = [c.id for c in default_ablations()]
        assert len(ids) == len(set(ids))


class TestAblationRunner:
    """Shell: ablation реально различает механизмы (VALIDATION §7.5)."""

    def test_policy_off_removes_reaction_classes(self, tmp_path: Path) -> None:
        """policy OFF → множество классов решений пусто (изменение есть)."""
        runner = BehavioralChainRunner(workdir=tmp_path)
        result = runner.run_ablation(
            AblationCheck(
                id="ablate.policy.decision",
                preset="dialogue",
                mechanism=Ablation.POLICY,
                observable="reaction_classes",
                messages=((0, "привет"),),
            )
        )
        assert result.passed
        assert result.changed
        assert result.baseline  # непусто с механизмом
        assert result.ablated == frozenset()

    def test_memory_off_changes_state(self, tmp_path: Path) -> None:
        """memory OFF целиком → числовое состояние (F) меняется."""
        runner = BehavioralChainRunner(workdir=tmp_path)
        result = runner.run_ablation(
            AblationCheck(
                id="ablate.memory.channel",
                preset="dialogue",
                mechanism=Ablation.MEMORY,
                observable="f_mean",
                messages=((0, "привет"),),
            )
        )
        assert result.passed
        assert result.changed
        assert result.baseline != result.ablated

    def test_recall_content_off_changes_state(self, tmp_path: Path) -> None:
        """Чистый ablation: recall OFF (prior молчит) → F меняется.

        Провайдер и размерность сохранены, так что различие F атрибутируется
        именно содержимому воспоминаний (не confounding канала).
        """
        runner = BehavioralChainRunner(workdir=tmp_path)
        result = runner.run_ablation(
            AblationCheck(
                id="ablate.recall.content",
                preset="dialogue",
                mechanism=Ablation.RECALL,
                observable="f_mean",
                messages=((0, "привет"), (60, "снова я")),
            )
        )
        assert result.passed
        assert result.changed
        assert result.baseline != result.ablated

    def test_expect_no_change_fails_when_changed(self, tmp_path: Path) -> None:
        """Ожидание «не изменилось», но механизм влияет → провал (анти-тавтология)."""
        runner = BehavioralChainRunner(workdir=tmp_path)
        result = runner.run_ablation(
            AblationCheck(
                id="ablate.recall.expect_none",
                preset="dialogue",
                mechanism=Ablation.RECALL,
                observable="f_mean",
                messages=((0, "привет"), (60, "снова я")),
                expect_change=False,
            )
        )
        assert not result.passed
        assert result.changed

    def test_determinism_same_seed(self, tmp_path: Path) -> None:
        """Одинаковый seed → одинаковый ablation-вердикт (VALIDATION §7.7)."""
        check = AblationCheck(
            id="ablate.recall.content",
            preset="dialogue",
            mechanism=Ablation.RECALL,
            observable="f_mean",
            messages=((0, "привет"), (60, "снова я")),
        )
        first = BehavioralChainRunner(workdir=tmp_path / "a").run_ablation(check)
        second = BehavioralChainRunner(workdir=tmp_path / "b").run_ablation(check)
        assert first.passed == second.passed
        assert first.baseline == second.baseline
        assert first.ablated == second.ablated

    def test_run_ablations_default_corpus(self, tmp_path: Path) -> None:
        results = BehavioralChainRunner(workdir=tmp_path).run_ablations()
        assert len(results) == len(default_ablations())
        assert all(r.passed for r in results)


class TestSummarize:
    """Сводка корпуса (Core)."""

    def test_empty(self) -> None:
        assert summarize([])["total"] == 0

    def test_counts_failures(self) -> None:
        ok = _result("a", passed=True)
        bad = _result("b", passed=False)
        summary = summarize([ok, bad])
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["failed_ids"] == ("b",)


class TestBehavioralChainRunner:
    """Shell: детерминированный прогон сценариев поверх HostLoop."""

    def test_state_scenario_passes(self, tmp_path: Path) -> None:
        runner = BehavioralChainRunner(workdir=tmp_path)
        result = runner.run(
            Scenario(id="state.bounds.baseline", link="state", preset="baseline")
        )
        assert result.passed
        assert result.violations == ()
        assert result.reason == "ok"

    def test_decision_respond_observed(self, tmp_path: Path) -> None:
        """Новое сообщение → класс RESPOND наблюдаем (звено 2)."""
        runner = BehavioralChainRunner(workdir=tmp_path)
        result = runner.run(
            Scenario(
                id="decision.respond",
                link="decision",
                preset="dialogue",
                messages=((0, "привет"),),
                expect_reaction=ReactionClass.RESPOND,
            )
        )
        assert result.passed
        assert ReactionClass.RESPOND in result.reactions

    def test_missing_expected_reaction_fails(self, tmp_path: Path) -> None:
        """Недостижимое ожидание → провал с внятной причиной (fail)."""
        runner = BehavioralChainRunner(workdir=tmp_path)
        result = runner.run(
            Scenario(
                id="decision.explore.unreachable",
                link="decision",
                preset="baseline",
                expect_reaction=ReactionClass.EXPLORE,
            )
        )
        assert not result.passed
        assert "expected reaction" in result.reason

    def test_determinism_same_seed(self, tmp_path: Path) -> None:
        """Одинаковый seed → одинаковые классы реакций (VALIDATION §7.7)."""
        scenario = Scenario(
            id="decision.respond",
            link="decision",
            preset="dialogue",
            messages=((0, "привет"),),
            expect_reaction=ReactionClass.RESPOND,
        )
        first = BehavioralChainRunner(workdir=tmp_path / "a").run(scenario)
        second = BehavioralChainRunner(workdir=tmp_path / "b").run(scenario)
        assert first.reactions == second.reactions

    def test_run_all_default_corpus(self, tmp_path: Path) -> None:
        results = BehavioralChainRunner(workdir=tmp_path).run_all()
        assert len(results) == len(default_scenarios())
        assert all(r.passed for r in results)


def _result(scenario_id: str, *, passed: bool):
    """Минимальный ScenarioResult для тестов сводки."""
    from src.host.behavioral_chain import ScenarioResult

    return ScenarioResult(
        scenario=Scenario(id=scenario_id, link="state", preset="baseline"),
        reactions=(),
        violations=() if passed else ("finite",),
        intent_violations=(),
        passed=passed,
        reason="ok" if passed else "state invariant violated: finite",
    )
