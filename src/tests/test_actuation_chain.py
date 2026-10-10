"""Tests for the sequenced-reaction chain harness (S8 stage 8, VALIDATION §7.10).

Covers the Core (form classification, chain invariants including the tool schema
and the argument-grounding fidelity axis, shares, scenario/ablation validation)
and the Shell ``ChainHarness``: a 2-step tool chain, a mixed speak+tool chain, the
generator ablation (the chain collapses — non-tautological) and determinism.
"""

from __future__ import annotations

import pytest

from src.core.actuation import (
    Actuation,
    ActuationKind,
    ActuationStatus,
    Effect,
    Fact,
    Option,
    OptionSource,
)
from src.host.behavioral_chain import (
    ArgumentGrounding,
    ChainAblation,
    ChainAblationCheck,
    ChainHarness,
    ChainInvariant,
    ChainScenario,
    ChainStep,
    ChainView,
    StepForm,
    chain_shares,
    check_chain,
    classify_argument,
    classify_step_form,
    default_chain_ablations,
    default_chain_scenarios,
)

_NET = Fact("report_ready")
_TOPIC = Fact("topic_bound")


def _tool_step(
    goal: str, *, status: ActuationStatus = ActuationStatus.SUCCESS
) -> ChainStep:
    """Собрать tool-шаг с данными."""
    return ChainStep(
        actuation=Actuation(ActuationKind.INVOKE_TOOL, goal, goal),
        status=status,
        gated=True,
        form=StepForm.TOOL,
        data=(0.0, 0.0),
    )


def _speak_step(
    goal: str, *, status: ActuationStatus = ActuationStatus.SUCCESS
) -> ChainStep:
    """Собрать речевой шаг с текстом."""
    return ChainStep(
        actuation=Actuation(ActuationKind.SPEAK, goal, "привет"),
        status=status,
        gated=True,
        form=StepForm.TEXT,
    )


def _view(
    steps: tuple[ChainStep, ...],
    *,
    completed: int | None = None,
    options: tuple[Option, ...] = (),
    expected_steps: int | None = None,
) -> ChainView:
    """Собрать снимок цепочки с безопасными дефолтами."""
    return ChainView(
        steps=steps,
        decided=len(steps),
        completed=(
            sum(1 for s in steps if s.status is ActuationStatus.SUCCESS)
            if completed is None
            else completed
        ),
        tool_steps=sum(
            1 for s in steps if s.actuation.kind is ActuationKind.INVOKE_TOOL
        ),
        options=options,
        expected_steps=expected_steps,
    )


class TestClassifyStepForm:
    """Форма результата шага (Core, звено 5, §7.10)."""

    def test_failure_is_empty(self) -> None:
        assert (
            classify_step_form(
                status=ActuationStatus.FAILURE, text="x", data=(1.0,)
            )
            is StepForm.EMPTY
        )

    def test_running_is_empty(self) -> None:
        assert (
            classify_step_form(
                status=ActuationStatus.RUNNING, text=None, data=()
            )
            is StepForm.EMPTY
        )

    def test_text_wins(self) -> None:
        assert (
            classify_step_form(
                status=ActuationStatus.SUCCESS, text="реплика", data=()
            )
            is StepForm.TEXT
        )

    def test_blank_text_is_not_text(self) -> None:
        """Пустой текст не считается речью; данные дают TOOL."""
        assert (
            classify_step_form(
                status=ActuationStatus.SUCCESS, text="   ", data=(1.0,)
            )
            is StepForm.TOOL
        )

    def test_data_is_tool(self) -> None:
        assert (
            classify_step_form(
                status=ActuationStatus.SUCCESS, text=None, data=(0.5, 0.5)
            )
            is StepForm.TOOL
        )

    def test_empty_success_is_empty(self) -> None:
        assert (
            classify_step_form(
                status=ActuationStatus.SUCCESS, text=None, data=()
            )
            is StepForm.EMPTY
        )


class TestClassifyArgument:
    """Grounding аргумента тула (Core, fidelity §7.10)."""

    def test_grounded_option_with_effect(self) -> None:
        options = (Option("tool:probe", OptionSource.TOOL, effect=Effect(_NET)),)
        assert classify_argument("tool:probe", options) is ArgumentGrounding.GROUNDED

    def test_epistemic_option_without_effect(self) -> None:
        options = (Option("tool:probe", OptionSource.TOOL),)
        assert classify_argument("tool:probe", options) is ArgumentGrounding.EPISTEMIC

    def test_unknown_name(self) -> None:
        options = (Option("tool:probe", OptionSource.TOOL, effect=Effect(_NET)),)
        assert classify_argument("tool:other", options) is ArgumentGrounding.UNKNOWN

    def test_empty_window(self) -> None:
        assert classify_argument("tool:probe", ()) is ArgumentGrounding.UNKNOWN


class TestCheckChain:
    """Инварианты цепочки актуаций (Core, звенья 4–5, §7.10)."""

    def test_clean_tool_chain_passes(self) -> None:
        options = (
            Option("tool:probe", OptionSource.TOOL, effect=Effect(_NET)),
            Option("tool:search", OptionSource.TOOL, effect=Effect(_TOPIC)),
        )
        view = _view(
            (_tool_step("tool:search"), _tool_step("tool:probe")),
            options=options,
            expected_steps=2,
        )
        assert check_chain(view, tuple(ChainInvariant)) == ()

    def test_preempted_step_fails_gated(self) -> None:
        step = ChainStep(
            actuation=Actuation(
                ActuationKind.INVOKE_TOOL, "tool:probe", "tool:probe"
            ),
            status=ActuationStatus.PREEMPTED,
            gated=False,
            form=StepForm.EMPTY,
        )
        view = _view((step,))
        assert "step_gated" in check_chain(view, (ChainInvariant.STEP_GATED,))

    def test_failed_step_fails_completed(self) -> None:
        view = _view((_tool_step("tool:probe", status=ActuationStatus.FAILURE),))
        assert "step_completed" in check_chain(
            view, (ChainInvariant.STEP_COMPLETED,)
        )

    def test_form_mismatch_fails(self) -> None:
        """Tool-актуация с текстовой формой → нарушение соответствия."""
        step = ChainStep(
            actuation=Actuation(
                ActuationKind.INVOKE_TOOL, "tool:probe", "tool:probe"
            ),
            status=ActuationStatus.SUCCESS,
            gated=True,
            form=StepForm.TEXT,
            data=(1.0,),
        )
        view = _view((step,))
        assert "form_matches_kind" in check_chain(
            view, (ChainInvariant.FORM_MATCHES_KIND,)
        )

    def test_empty_form_is_tolerated(self) -> None:
        """EMPTY не противоречит виду (сбой честен, не подмена формы)."""
        step = ChainStep(
            actuation=Actuation(
                ActuationKind.INVOKE_TOOL, "tool:probe", "tool:probe"
            ),
            status=ActuationStatus.FAILURE,
            gated=True,
            form=StepForm.EMPTY,
        )
        view = _view((step,))
        assert "form_matches_kind" not in check_chain(
            view, (ChainInvariant.FORM_MATCHES_KIND,)
        )

    def test_schema_dim_mismatch_fails(self) -> None:
        """Данные тула не совпали с ожидаемой схемой (длиной)."""
        step = ChainStep(
            actuation=Actuation(
                ActuationKind.INVOKE_TOOL, "tool:probe", "tool:probe"
            ),
            status=ActuationStatus.SUCCESS,
            gated=True,
            form=StepForm.TOOL,
            data=(0.0, 0.0),
            expected_dim=3,
        )
        view = _view((step,))
        assert "schema_matches_dim" in check_chain(
            view, (ChainInvariant.SCHEMA_MATCHES_DIM,)
        )

    def test_schema_dim_match_passes(self) -> None:
        step = ChainStep(
            actuation=Actuation(
                ActuationKind.INVOKE_TOOL, "tool:probe", "tool:probe"
            ),
            status=ActuationStatus.SUCCESS,
            gated=True,
            form=StepForm.TOOL,
            data=(0.0, 0.0),
            expected_dim=2,
        )
        view = _view((step,))
        assert check_chain(view, (ChainInvariant.SCHEMA_MATCHES_DIM,)) == ()

    def test_schema_undeclared_is_skipped(self) -> None:
        """Без объявленной схемы (expected_dim=None) проверка не срабатывает."""
        view = _view((_tool_step("tool:probe"),))
        assert check_chain(view, (ChainInvariant.SCHEMA_MATCHES_DIM,)) == ()

    def test_step_count_mismatch_fails(self) -> None:
        view = _view((_tool_step("tool:probe"),), expected_steps=2)
        assert "step_count" in check_chain(view, (ChainInvariant.STEP_COUNT,))

    def test_step_count_match_passes(self) -> None:
        view = _view((_tool_step("tool:probe"),), expected_steps=1)
        assert check_chain(view, (ChainInvariant.STEP_COUNT,)) == ()

    def test_step_count_undeclared_is_skipped(self) -> None:
        view = _view((_tool_step("tool:probe"),))
        assert check_chain(view, (ChainInvariant.STEP_COUNT,)) == ()

    def test_argument_not_grounded_fails(self) -> None:
        """Имя тула не из окна → аргумент не следует из подцели."""
        view = _view((_tool_step("tool:ghost"),))
        assert "argument_grounded" in check_chain(
            view, (ChainInvariant.ARGUMENT_GROUNDED,)
        )

    def test_argument_grounded_passes(self) -> None:
        options = (Option("tool:probe", OptionSource.TOOL, effect=Effect(_NET)),)
        view = _view((_tool_step("tool:probe"),), options=options)
        assert check_chain(view, (ChainInvariant.ARGUMENT_GROUNDED,)) == ()

    def test_speak_step_needs_no_argument(self) -> None:
        """Речевой шаг не проверяется на grounding аргумента (не тул)."""
        view = _view((_speak_step("say:topic"),))
        assert check_chain(view, (ChainInvariant.ARGUMENT_GROUNDED,)) == ()

    def test_only_requested_invariants_checked(self) -> None:
        view = _view((_tool_step("tool:ghost"),))
        assert check_chain(view, (ChainInvariant.STEP_GATED,)) == ()


class TestChainShares:
    """Доли наблюдаемых цепочки (Core, §7.10)."""

    def test_empty_is_zero(self) -> None:
        assert chain_shares(_view(())) == {"multi_step": 0.0, "tool": 0.0}

    def test_single_tool_step(self) -> None:
        assert chain_shares(_view((_tool_step("tool:probe"),))) == {
            "multi_step": 0.0,
            "tool": 1.0,
        }

    def test_multi_step_tool(self) -> None:
        view = _view((_tool_step("a"), _tool_step("b")))
        assert chain_shares(view) == {"multi_step": 1.0, "tool": 1.0}

    def test_mixed_tool_fraction(self) -> None:
        view = _view((_speak_step("say:x"), _tool_step("tool:probe")))
        shares = chain_shares(view)
        assert shares["multi_step"] == 1.0
        assert shares["tool"] == pytest.approx(0.5)


class TestChainScenarioValidation:
    """Валидация ячейки сложной реакции (Core, §7.10)."""

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="id must not be empty"):
            ChainScenario(id="", goal_fact="x")

    def test_empty_goal_fact_raises(self) -> None:
        with pytest.raises(ValueError, match="goal_fact must not be empty"):
            ChainScenario(id="x", goal_fact="")

    def test_bad_tool_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="tool dim must be >= 1"):
            ChainScenario(id="x", goal_fact="y", tool_dims={"tool:a": 0})

    def test_goal_from_fact(self) -> None:
        scenario = ChainScenario(id="x", goal_fact="report_ready", goal_value=0.5)
        goal = scenario.goal()
        assert goal.fact.name == "report_ready"
        assert goal.value == 0.5


class TestChainAblationCheck:
    """Валидация проверки атрибуции генератора (Core, §7.10)."""

    def test_empty_id_raises(self) -> None:
        scenario = ChainScenario(id="x", goal_fact="y")
        with pytest.raises(ValueError, match="ablation id must not be empty"):
            ChainAblationCheck(id="", scenario=scenario)

    def test_default_mechanism_is_generator(self) -> None:
        scenario = ChainScenario(id="x", goal_fact="y")
        check = ChainAblationCheck(id="a", scenario=scenario)
        assert check.mechanism is ChainAblation.GENERATOR
        assert check.expect_change is True


class TestChainHarness:
    """Shell: прогон сложной реакции через ActuatorExecutor (§7.10)."""

    def test_two_step_tool_chain(self) -> None:
        scenario = next(
            s for s in default_chain_scenarios() if s.id == "chain.two_step.tool"
        )
        view = ChainHarness().run(scenario)
        assert view.decided == 2
        assert view.completed == 2
        assert view.multi_step
        assert view.tool_steps == 2
        assert [s.form for s in view.steps] == [StepForm.TOOL, StepForm.TOOL]
        assert check_chain(view, tuple(ChainInvariant)) == ()

    def test_one_step_tool_chain(self) -> None:
        scenario = next(
            s for s in default_chain_scenarios() if s.id == "chain.one_step.tool"
        )
        view = ChainHarness().run(scenario)
        assert view.decided == 1
        assert not view.multi_step
        assert view.tool_steps == 1
        assert check_chain(view, tuple(ChainInvariant)) == ()

    def test_mixed_speak_and_tool_chain(self) -> None:
        """Смешанная цепочка покрывает и текстовую форму (§7.10)."""
        scenario = next(
            s for s in default_chain_scenarios() if s.id == "chain.two_step.mixed"
        )
        view = ChainHarness().run(scenario)
        assert view.decided == 2
        forms = [s.form for s in view.steps]
        assert StepForm.TEXT in forms
        assert StepForm.TOOL in forms
        assert check_chain(view, tuple(ChainInvariant)) == ()

    def test_default_corpus_all_pass(self) -> None:
        harness = ChainHarness()
        for scenario in default_chain_scenarios():
            view = harness.run(scenario)
            assert check_chain(view, tuple(ChainInvariant)) == ()
            assert view.decided == scenario.expected_steps

    def test_max_ticks_validation(self) -> None:
        with pytest.raises(ValueError, match="max_ticks must be >= 1"):
            ChainHarness(max_ticks=0)

    def test_max_depth_truncates_chain(self) -> None:
        """Малый горизонт генератора обрезает цепочку (горизонт §7.10).

        При ``max_depth=1`` подцель не разворачивается в действие (безопасный
        отказ) — цепочка короче полной двухшаговой.
        """
        scenario = next(
            s for s in default_chain_scenarios() if s.id == "chain.two_step.tool"
        )
        full = ChainHarness().run(scenario)
        shallow = ChainHarness().run(scenario, max_depth=1)
        assert full.decided == 2
        assert shallow.decided < full.decided

    def test_determinism_same_input(self) -> None:
        """Одинаковый вход → одинаковый ChainView (§7.7)."""
        scenario = next(
            s for s in default_chain_scenarios() if s.id == "chain.two_step.tool"
        )
        first = ChainHarness().run(scenario)
        second = ChainHarness().run(scenario)
        assert first == second

    def test_speak_override_is_used(self) -> None:
        """Инъекция речи: текст шага берётся из переданной функции."""
        scenario = next(
            s for s in default_chain_scenarios() if s.id == "chain.two_step.mixed"
        )
        view = ChainHarness().run(scenario, speak=lambda _a: "кастом")
        text_step = next(s for s in view.steps if s.form is StepForm.TEXT)
        assert text_step.actuation.kind is ActuationKind.SPEAK


class TestChainAblation:
    """Shell: ablation генератора схлопывает цепочку (анти-тавтология, §7.10)."""

    def test_generator_ablation_collapses(self) -> None:
        harness = ChainHarness()
        result = harness.run_ablation(default_chain_ablations()[0])
        assert result.passed
        assert result.changed
        assert result.baseline_steps == 2
        assert result.ablated_steps == 0

    def test_ablation_expect_no_change_fails(self) -> None:
        """Ожидание «не изменилось», но генератор влияет → провал."""
        scenario = next(
            s for s in default_chain_scenarios() if s.id == "chain.two_step.tool"
        )
        check = ChainAblationCheck(
            id="ablate.chain.expect_none",
            scenario=scenario,
            mechanism=ChainAblation.GENERATOR,
            expect_change=False,
        )
        result = ChainHarness().run_ablation(check)
        assert not result.passed
        assert result.changed

    def test_ablation_determinism(self) -> None:
        check = default_chain_ablations()[0]
        first = ChainHarness().run_ablation(check)
        second = ChainHarness().run_ablation(check)
        assert first.baseline_steps == second.baseline_steps
        assert first.ablated_steps == second.ablated_steps
        assert first.passed == second.passed


class TestDefaultCorpus:
    """Встроенный корпус сложных реакций (§7.10)."""

    def test_ids_unique(self) -> None:
        ids = [s.id for s in default_chain_scenarios()]
        assert len(ids) == len(set(ids))

    def test_corpus_covers_text_and_tool_forms(self) -> None:
        forms: set[StepForm] = set()
        harness = ChainHarness()
        for scenario in default_chain_scenarios():
            forms.update(s.form for s in harness.run(scenario).steps)
        assert StepForm.TEXT in forms
        assert StepForm.TOOL in forms

    def test_ablation_ids_unique(self) -> None:
        ids = [c.id for c in default_chain_ablations()]
        assert len(ids) == len(set(ids))

    def test_ablation_scenario_is_from_corpus(self) -> None:
        corpus_ids = {s.id for s in default_chain_scenarios()}
        assert all(c.scenario.id in corpus_ids for c in default_chain_ablations())
