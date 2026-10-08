"""Tests for the behavioral chain harness Core + Shell (VALIDATION §7)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.core.policy import Action, PolicyCandidate, PolicyTrace
from src.host.behavioral_chain import (
    Ablation,
    AblationCheck,
    ActuationInvariant,
    ActuationView,
    BehavioralChainRunner,
    DistortionClass,
    EmbeddingToneScorer,
    FailingLlmClient,
    FidelityHarness,
    FidelityPair,
    IntentInvariant,
    IntentView,
    LexiconToneScorer,
    Precondition,
    PreconditionKind,
    ReactionClass,
    RecordingLlmClient,
    ReplyClass,
    ReplyInvariant,
    ReplyView,
    Scenario,
    StateInvariant,
    StateView,
    ToneAxis,
    check_actuation,
    check_fidelity,
    check_intent,
    check_reply,
    check_state,
    classify_reaction,
    classify_reply,
    default_ablations,
    default_fidelity_pairs,
    default_scenarios,
    intent_from_state,
    llm_responder,
    report_dict,
    result_dict,
    summarize,
)
from src.speech.controller import SpeechDecision
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

    def test_primed_without_measured_message_raises(self) -> None:
        """primed(n) без сообщения после прогрева не декларируется (§7.8)."""
        with pytest.raises(ValueError):
            Scenario(
                id="x",
                link="state",
                preset="dialogue",
                messages=((0, "a"),),
                precondition=Precondition.primed(1),
            )

    def test_measure_from_born_is_zero(self) -> None:
        assert (
            Scenario(id="x", link="state", preset="baseline").measure_from() == 0
        )

    def test_measure_from_matured_is_zero(self) -> None:
        scenario = Scenario(
            id="x",
            link="state",
            preset="long-horizon",
            ticks=300,
            precondition=Precondition.matured(),
        )
        assert scenario.measure_from() == 0

    def test_measure_from_primed_is_next_message(self) -> None:
        """Первые n сообщений — прогрев; замер с (n+1)-го (§7.8)."""
        scenario = Scenario(
            id="x",
            link="state",
            preset="dialogue",
            messages=((0, "a"), (40, "b"), (80, "c")),
            precondition=Precondition.primed(2),
        )
        assert scenario.measure_from() == 80


class TestPrecondition:
    """Предусловия born/primed/matured (Core; VALIDATION §7.8)."""

    def test_born_default(self) -> None:
        pre = Precondition.born()
        assert pre.kind is PreconditionKind.BORN
        assert pre.warmup == 0
        assert str(pre) == "born"

    def test_primed_carries_warmup(self) -> None:
        pre = Precondition.primed(3)
        assert pre.kind is PreconditionKind.PRIMED
        assert pre.warmup == 3
        assert str(pre) == "primed(3)"

    def test_matured(self) -> None:
        pre = Precondition.matured()
        assert pre.kind is PreconditionKind.MATURED
        assert str(pre) == "matured"

    def test_primed_requires_warmup(self) -> None:
        with pytest.raises(ValueError):
            Precondition.primed(0)

    def test_negative_warmup_raises(self) -> None:
        with pytest.raises(ValueError):
            Precondition(kind=PreconditionKind.PRIMED, warmup=-1)

    def test_warmup_outside_primed_raises(self) -> None:
        with pytest.raises(ValueError):
            Precondition(kind=PreconditionKind.BORN, warmup=2)
        with pytest.raises(ValueError):
            Precondition(kind=PreconditionKind.MATURED, warmup=1)

    def test_applies_to_exact_kind(self) -> None:
        assert Precondition.born().applies_to(Precondition.born())
        assert not Precondition.born().applies_to(Precondition.matured())
        assert Precondition.matured().applies_to(Precondition.matured())
        assert not Precondition.matured().applies_to(Precondition.born())

    def test_applies_to_primed_volume(self) -> None:
        """primed(n) применим, если прогон даёт не меньше прогрева."""
        assert Precondition.primed(1).applies_to(Precondition.primed(2))
        assert Precondition.primed(2).applies_to(Precondition.primed(2))
        assert not Precondition.primed(3).applies_to(Precondition.primed(2))


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
    """Встроенный корпус звеньев 1–5."""

    def test_covers_state_decisions_intent(self) -> None:
        links = {s.link for s in default_scenarios()}
        assert links == {"state", "decision", "intent", "actuation", "reply"}

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


class TestActuationCheck:
    """Инварианты актюации (Core, звено 4)."""

    def _view(self, **overrides: object) -> ActuationView:
        base: dict[str, object] = {
            "decision": SpeechDecision(speak=True, reason="respond"),
            "called": True,
            "response": "Понял, отвечаю.",
            "error": False,
            "frame_goal": "respond",
            "action": Action.RESPOND,
            "escape_hatch": False,
        }
        base.update(overrides)
        return ActuationView(**base)  # type: ignore[arg-type]

    def test_clean_passes(self) -> None:
        assert check_actuation(self._view(), tuple(ActuationInvariant)) == ()

    def test_silent_with_call_fails(self) -> None:
        view = self._view(
            decision=SpeechDecision(speak=False, reason="silent"),
            called=True,
        )
        assert "llm_called_iff_speak" in check_actuation(
            view, (ActuationInvariant.LLM_CALLED_IFF_SPEAK,)
        )

    def test_speak_without_call_fails(self) -> None:
        view = self._view(called=False, response=None)
        assert "llm_called_iff_speak" in check_actuation(
            view, (ActuationInvariant.LLM_CALLED_IFF_SPEAK,)
        )

    def test_escape_hatch_exempt_from_call(self) -> None:
        """Escape hatch — дешёвый путь: говорение без вызова LLM допустимо."""
        view = self._view(called=False, response="Нагрузка высокая", escape_hatch=True)
        assert check_actuation(view, tuple(ActuationInvariant)) == ()

    def test_error_does_not_violate_response_returned(self) -> None:
        view = self._view(called=True, response=None, error=True)
        assert "response_returned" not in check_actuation(
            view, (ActuationInvariant.RESPONSE_RETURNED,)
        )

    def test_frame_goal_mismatch_fails(self) -> None:
        view = self._view(frame_goal="explore", action=Action.RESPOND)
        assert "frame_grounded" in check_actuation(
            view, (ActuationInvariant.FRAME_GROUNDED,)
        )


class TestClassifyReply:
    """Структурная классификация реплики (Core, звено 5)."""

    def test_empty(self) -> None:
        assert classify_reply(None) is ReplyClass.EMPTY
        assert classify_reply("   ") is ReplyClass.EMPTY

    def test_question(self) -> None:
        assert classify_reply("Как тебя зовут?") is ReplyClass.QUESTION

    def test_statement(self) -> None:
        assert classify_reply("Понял тебя.") is ReplyClass.STATEMENT


class TestCheckReply:
    """Структурная релевантность реплики интенту (Core, звено 5)."""

    def _frame(self, goal: str):
        return build_intent_frame(
            f=1.0, valence=0.0, stress=0.0, task="none", goal=goal
        )

    def test_statement_ok_for_respond(self) -> None:
        view = ReplyView(frame=self._frame("respond"), text="Понял тебя.")
        assert check_reply(view, tuple(ReplyInvariant)) == ()

    def test_statement_fails_for_explore(self) -> None:
        view = ReplyView(frame=self._frame("explore"), text="Понял тебя.")
        assert "class_matches_goal" in check_reply(
            view, (ReplyInvariant.CLASS_MATCHES_GOAL,)
        )

    def test_question_ok_for_identify(self) -> None:
        view = ReplyView(frame=self._frame("identify_partner"), text="Как звать?")
        assert check_reply(view, (ReplyInvariant.CLASS_MATCHES_GOAL,)) == ()

    def test_empty_fails_nonempty(self) -> None:
        view = ReplyView(frame=self._frame("respond"), text=None)
        assert "nonempty" in check_reply(view, (ReplyInvariant.NONEMPTY,))


class TestRecordingLlmClient:
    """Fake-LLM звеньев 4–5 (Shell)."""

    def test_records_calls(self) -> None:
        client = RecordingLlmClient()
        assert client.call_count == 0
        client.reply([{"role": "system", "content": "x"}])
        assert client.call_count == 1

    def test_question_for_identify_goal(self) -> None:
        from src.speech.intent import goal_instruction

        client = RecordingLlmClient()
        system = goal_instruction("identify_partner")
        reply = client.reply([{"role": "system", "content": system}])
        assert classify_reply(reply) is ReplyClass.QUESTION

    def test_question_for_explore_goal(self) -> None:
        from src.speech.intent import goal_instruction

        client = RecordingLlmClient()
        system = goal_instruction("explore")
        reply = client.reply([{"role": "system", "content": system}])
        assert classify_reply(reply) is ReplyClass.QUESTION

    def test_statement_for_respond_goal(self) -> None:
        from src.speech.intent import goal_instruction

        client = RecordingLlmClient()
        system = goal_instruction("respond")
        reply = client.reply([{"role": "system", "content": system}])
        assert classify_reply(reply) is ReplyClass.STATEMENT


class TestActuationRunner:
    """Shell: звенья 4–5 прогоняются поверх HostLoop."""

    def test_respond_scenario_calls_llm(self, tmp_path: Path) -> None:
        runner = BehavioralChainRunner(workdir=tmp_path)
        result = runner.run(
            Scenario(
                id="actuation.respond.calls_llm",
                link="actuation",
                preset="dialogue",
                messages=((0, "привет"),),
            )
        )
        assert result.passed
        assert result.llm_calls >= 1

    def test_baseline_invariants_pass(self, tmp_path: Path) -> None:
        """Инвариант «LLM вызван ⇔ решение говорить» держится и без сообщения."""
        runner = BehavioralChainRunner(workdir=tmp_path)
        result = runner.run(
            Scenario(
                id="actuation.baseline.invariants",
                link="actuation",
                preset="baseline",
            )
        )
        assert result.passed

    def test_llm_failure_is_graceful(self, tmp_path: Path) -> None:
        """Сбой LLM не роняет прогон: ответ None, звено 4 помечает сбой."""
        runner = BehavioralChainRunner(
            workdir=tmp_path, llm_factory=FailingLlmClient
        )
        result = runner.run(
            Scenario(
                id="actuation.respond.failure",
                link="actuation",
                preset="dialogue",
                messages=((0, "привет"),),
            )
        )
        assert result.passed
        assert result.llm_calls >= 1

    def test_reply_scenario_passes(self, tmp_path: Path) -> None:
        runner = BehavioralChainRunner(workdir=tmp_path)
        result = runner.run(
            Scenario(
                id="reply.respond.class_matches",
                link="reply",
                preset="dialogue",
                messages=((0, "привет"),),
            )
        )
        assert result.passed
        assert result.reply_violations == ()

    def test_determinism_same_seed(self, tmp_path: Path) -> None:
        scenario = Scenario(
            id="reply.respond.class_matches",
            link="reply",
            preset="dialogue",
            messages=((0, "привет"),),
        )
        first = BehavioralChainRunner(workdir=tmp_path / "a").run(scenario)
        second = BehavioralChainRunner(workdir=tmp_path / "b").run(scenario)
        assert first.llm_calls == second.llm_calls
        assert first.reactions == second.reactions


class TestLexiconToneScorer:
    """Детерминированный скорер тона (Core, fidelity)."""

    def test_valence_order(self) -> None:
        scorer = LexiconToneScorer()
        assert scorer.score("мне плохо и тяжело", ToneAxis.VALENCE) < scorer.score(
            "мне хорошо и радостно", ToneAxis.VALENCE
        )

    def test_stress_counts(self) -> None:
        scorer = LexiconToneScorer()
        assert scorer.score("спокойно", ToneAxis.STRESS) < scorer.score(
            "мне тяжело и срочно!", ToneAxis.STRESS
        )

    def test_scope_question(self) -> None:
        scorer = LexiconToneScorer()
        assert scorer.score("Как дела?", ToneAxis.GOAL_SCOPE) == 1.0
        assert scorer.score("Понял.", ToneAxis.GOAL_SCOPE) == 0.0


class TestCheckFidelity:
    """Проверка порядка сигнала (Core, VALIDATION §7.6)."""

    def _pair(self, *, expect_ordered: bool = True) -> FidelityPair:
        return FidelityPair(
            id="fidelity.test",
            frame_a=build_intent_frame(
                f=1.0, valence=-3.0, stress=0.0, task="none", goal="respond"
            ),
            frame_b=build_intent_frame(
                f=1.0, valence=3.0, stress=0.0, task="none", goal="respond"
            ),
            axis=ToneAxis.VALENCE,
            expect_ordered=expect_ordered,
        )

    def test_order_preserved(self) -> None:
        result = check_fidelity(self._pair(), score_a=0.0, score_b=1.0)
        assert result.passed
        assert result.ordered
        assert result.distortion is DistortionClass.NONE

    def test_inversion_detected(self) -> None:
        result = check_fidelity(self._pair(), score_a=1.0, score_b=0.0)
        assert not result.passed
        assert result.distortion is DistortionClass.INVERSION

    def test_masking_detected(self) -> None:
        result = check_fidelity(self._pair(), score_a=0.5, score_b=0.5)
        assert result.distortion is DistortionClass.MASKING

    def test_non_finite_fails(self) -> None:
        result = check_fidelity(
            self._pair(), score_a=float("nan"), score_b=1.0
        )
        assert not result.passed
        assert result.distortion is DistortionClass.MASKING

    def test_fabrication_detected_when_signal_absent(self) -> None:
        """Ожидания сигнала нет: добавленный тон — фабрикация."""
        pair = FidelityPair(
            id="fidelity.no_signal",
            frame_a=build_intent_frame(
                f=1.0, valence=0.0, stress=0.0, task="none", goal="respond"
            ),
            frame_b=build_intent_frame(
                f=1.0, valence=0.0, stress=0.0, task="none", goal="respond"
            ),
            axis=ToneAxis.VALENCE,
            expect_ordered=False,
        )
        fabricated = check_fidelity(pair, score_a=0.0, score_b=2.0)
        assert not fabricated.passed
        assert fabricated.distortion is DistortionClass.FABRICATION
        faithful = check_fidelity(pair, score_a=0.0, score_b=0.0)
        assert faithful.passed
        assert faithful.distortion is DistortionClass.NONE


class TestFidelityHarness:
    """Shell: fidelity-прогон с детерминированным responder."""

    @staticmethod
    def _faithful(frame) -> str:
        """Детерминированный responder, сохраняющий сигнал."""
        if frame.goal in ("identify_partner", "explore"):
            return "Уточни, пожалуйста?"
        if frame.valence < -1.0:
            return "мне плохо и тяжело"
        if frame.valence > 1.0:
            return "мне хорошо и радостно"
        if frame.stress > 5.0:
            return "мне тяжело и срочно!"
        return "понял"

    def test_default_pairs_pass_with_faithful_responder(self) -> None:
        harness = FidelityHarness(self._faithful)
        results = harness.run_all()
        assert results
        assert all(r.passed for r in results)

    def test_inverting_responder_fails(self) -> None:
        def inverting(frame) -> str:
            # Инверсия: высокий valence → негативный тон.
            if frame.valence > 1.0:
                return "мне плохо и тяжело"
            if frame.valence < -1.0:
                return "мне хорошо и радостно"
            return "понял"

        harness = FidelityHarness(inverting)
        results = harness.run_all()
        valence = [r for r in results if r.pair.axis is ToneAxis.VALENCE]
        assert valence and not all(r.passed for r in valence)

    def test_default_pairs_corpus(self) -> None:
        pairs = default_fidelity_pairs()
        axes = {p.axis for p in pairs}
        assert axes == {ToneAxis.VALENCE, ToneAxis.STRESS, ToneAxis.GOAL_SCOPE}
        assert any(not p.expect_ordered for p in pairs)  # контроль фабрикации

    def test_embedding_scorer_opt_in(self) -> None:
        """Opt-in скорер: использует реальный/fake Embedder, не сеть в CI."""
        from src.memory.embedder import FakeEmbedder

        scorer = EmbeddingToneScorer(embedder=FakeEmbedder(dim=32))
        harness = FidelityHarness(self._faithful, scorer=scorer)
        # Детерминированный fake-эмбеддер: порядок по scope сохранён.
        scope = [
            r for r in harness.run_all() if r.pair.axis is ToneAxis.GOAL_SCOPE
        ]
        assert scope and all(r.passed for r in scope)

    @pytest.mark.skipif(
        not (os.environ.get("LLM_API_KEY") or os.environ.get("EMBEDDER_API_KEY")),
        reason="opt-in: real LLM fidelity run requires an API key",
    )
    def test_real_llm_opt_in(self) -> None:
        """Opt-in прогон с реальной LLM: порядок сигнала, не абсолют (§7.6).

        Не входит в CI-гейт: сеть и недетерминизм. Проверяет, что монотонность
        держится хотя бы по scope (структурная ось, устойчива к формулировке).
        """
        from src.speech.llm import build_llm_client

        llm = build_llm_client(mode="api")
        harness = FidelityHarness(llm_responder(llm))
        results = harness.run_all()
        scope = [r for r in results if r.pair.axis is ToneAxis.GOAL_SCOPE]
        assert scope and all(r.passed for r in scope)


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


class TestReport:
    """Сериализуемый отчёт прогона (Core; CLI/JSON)."""

    def test_result_dict_fields(self) -> None:
        result = _result("a", passed=True)
        data = result_dict(result)
        assert data["id"] == "a"
        assert data["passed"] is True
        assert data["precondition"] == "born"
        assert data["reactions"] == []
        assert data["state_violations"] == []

    def test_report_dict_summary_and_scenarios(self) -> None:
        ok = _result("a", passed=True)
        bad = _result("b", passed=False)
        report = report_dict([ok, bad])
        assert report["summary"]["total"] == 2
        assert report["summary"]["failed_ids"] == ["b"]
        assert [s["id"] for s in report["scenarios"]] == ["a", "b"]

    def test_report_dict_is_json_serializable(self) -> None:
        import json

        report = report_dict([_result("a", passed=False)])
        text = json.dumps(report, ensure_ascii=False)
        assert '"failed_ids": ["a"]' in text


class TestBehavioralCli:
    """CLI-вход b-теста (S7; --behavioral)."""

    def test_format_reactions_counts(self) -> None:
        from src.__main__ import _format_reactions

        reactions = (
            ReactionClass.SILENT,
            ReactionClass.SILENT,
            ReactionClass.RESPOND,
        )
        assert _format_reactions(reactions) == "respond×1, silent×2"

    def test_format_reactions_empty(self) -> None:
        from src.__main__ import _format_reactions

        assert _format_reactions(()) == "-"

    def test_parse_args_defaults(self) -> None:
        from src.__main__ import _parse_args

        args = _parse_args([])
        assert args.behavioral is False
        assert args.behavioral_precondition is None
        assert args.behavioral_warmup == 1
        assert args.behavioral_json is None

    def test_parse_args_flags(self) -> None:
        from src.__main__ import _parse_args

        args = _parse_args(
            [
                "--behavioral",
                "--behavioral-precondition",
                "primed",
                "--behavioral-warmup",
                "3",
                "--behavioral-json",
                "out.json",
            ]
        )
        assert args.behavioral is True
        assert args.behavioral_precondition == "primed"
        assert args.behavioral_warmup == 3
        assert args.behavioral_json == Path("out.json")

    def test_run_behavioral_writes_json_and_passes(self, tmp_path: Path) -> None:
        """CLI-прогон born-корпуса: код 0 и сериализуемый отчёт."""
        import json

        from src.__main__ import _parse_args, _run_behavioral

        report_path = tmp_path / "report.json"
        args = _parse_args(
            [
                "--behavioral",
                "--behavioral-precondition",
                "born",
                "--behavioral-json",
                str(report_path),
            ]
        )
        code = _run_behavioral(args)
        assert code == 0
        report = json.loads(report_path.read_text())
        assert report["summary"]["failed"] == 0
        assert report["scenarios"]
        assert all(
            s["precondition"] == "born" for s in report["scenarios"]
        )


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


class TestPreconditionRunner:
    """Shell: фильтрация применимых и прогрев primed (VALIDATION §7.8)."""

    def test_default_corpus_declares_all_preconditions(self) -> None:
        kinds = {s.precondition.kind for s in default_scenarios()}
        assert kinds == {
            PreconditionKind.BORN,
            PreconditionKind.PRIMED,
            PreconditionKind.MATURED,
        }

    def test_born_run_selects_only_born(self, tmp_path: Path) -> None:
        runner = BehavioralChainRunner(workdir=tmp_path)
        results = runner.run_all(precondition=Precondition.born())
        assert results
        assert all(
            r.scenario.precondition.kind is PreconditionKind.BORN
            for r in results
        )
        assert all(r.passed for r in results)

    def test_primed_run_selects_primed(self, tmp_path: Path) -> None:
        runner = BehavioralChainRunner(workdir=tmp_path)
        results = runner.run_all(precondition=Precondition.primed(1))
        assert results
        assert all(
            r.scenario.precondition.kind is PreconditionKind.PRIMED
            for r in results
        )
        assert all(r.passed for r in results)

    def test_primed_warmup_excludes_early_ticks(self, tmp_path: Path) -> None:
        """Прогрев не записывается: наблюдаемые только с измеряемого тика."""
        runner = BehavioralChainRunner(workdir=tmp_path)
        scenario = Scenario(
            id="primed.probe",
            link="decision",
            preset="dialogue",
            ticks=120,
            messages=((0, "a"), (40, "b")),
            precondition=Precondition.primed(1),
        )
        result = runner.run(scenario)
        assert len(result.reactions) == 120 - 40

    def test_run_matured_only_matured(self, tmp_path: Path) -> None:
        runner = BehavioralChainRunner(workdir=tmp_path)
        results = runner.run_matured()
        assert results
        assert all(
            r.scenario.precondition.kind is PreconditionKind.MATURED
            for r in results
        )
        assert all(r.passed for r in results)


def _result(scenario_id: str, *, passed: bool):
    """Минимальный ScenarioResult для тестов сводки."""
    from src.host.behavioral_chain import ScenarioResult

    return ScenarioResult(
        scenario=Scenario(id=scenario_id, link="state", preset="baseline"),
        reactions=(),
        violations=() if passed else ("finite",),
        intent_violations=(),
        actuation_violations=(),
        reply_violations=(),
        llm_calls=0,
        passed=passed,
        reason="ok" if passed else "state invariant violated: finite",
    )
