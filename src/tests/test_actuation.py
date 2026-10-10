"""Unit tests for actuation Functional Core (S8 stage 1, ADR-0012).

Open option window, total scorer, deterministic choice. Covers openness (a
new tool is visible with no code for it), totality (unknown options degrade to
the epistemic default), visibility-not-authorization (irreversible options are
present and down-weighted), determinism (tie-break by window order) and input
validation.
"""

from __future__ import annotations

import pytest

from src.core.actuation import (
    ActuationPreferences,
    Option,
    OptionContext,
    OptionSource,
    OptionWindow,
    build_options,
    score_option,
    select_option,
)
from src.mcp.models import SignalCategory
from src.mcp.probe import Affordance


def _affordance(name: str = "web_search", *, reversible: bool = True) -> Affordance:
    """Собрать аффорданс для теста."""
    return Affordance(name, SignalCategory.EXTEROCEPTIVE, reversible)


def _context(uncertainty: float = 0.8) -> OptionContext:
    """Собрать контекст опций для теста."""
    return OptionContext(uncertainty=uncertainty)


class TestBuildOptions:
    """Tests for build_options — openness of the window."""

    def test_tools_become_options_without_code(self) -> None:
        """Новый тул виден в окне без кода под него (открытость)."""
        window = build_options([_affordance("get_weather"), _affordance("web_search")])

        assert window.ids == ("tool:get_weather", "tool:web_search")
        assert all(o.source is OptionSource.TOOL for o in window.options)

    def test_order_is_input_order(self) -> None:
        """Порядок окна = порядок входа (детерминизм/тай-брейк)."""
        window = build_options([_affordance("b"), _affordance("a")])

        assert window.ids == ("tool:b", "tool:a")

    def test_target_is_raw_affordance_name(self) -> None:
        """id неймспейснут (tool:<name>), target = реальное имя тула."""
        window = build_options([_affordance("get_weather"), _affordance("web_search")])

        assert [o.target for o in window.options] == ["get_weather", "web_search"]

    def test_reversibility_carried_over(self) -> None:
        """Reversible аффорданса переносится в опцию."""
        window = build_options([_affordance("send_mail", reversible=False)])

        option = window.find("tool:send_mail")
        assert option is not None
        assert option.reversible is False

    def test_descriptions_enrichment(self) -> None:
        """descriptions — аддитивное обогащение; без него пусто (дефолт)."""
        window = build_options(
            [_affordance("web_search")],
            descriptions={"web_search": "search the web"},
        )

        option = window.find("tool:web_search")
        assert option is not None
        assert option.description == "search the web"
        # без обогащения — работоспособный дефолт
        bare = build_options([_affordance("web_search")])
        bare_option = bare.find("tool:web_search")
        assert bare_option is not None
        assert bare_option.description == ""

    def test_builtin_appended_after_tools(self) -> None:
        """Builtin-опции идут после тулов (стабильный порядок)."""
        builtin = Option("builtin:speak", OptionSource.BUILTIN, reversible=True)
        window = build_options([_affordance()], builtin=[builtin])

        assert window.ids == ("tool:web_search", "builtin:speak")
        assert window.tools == (build_options([_affordance()]).options[0],)

    def test_duplicate_id_rejected(self) -> None:
        """Дубликат id в окне → ValueError."""
        with pytest.raises(ValueError, match="duplicate option id"):
            OptionWindow(
                (Option("x", OptionSource.TOOL), Option("x", OptionSource.TOOL))
            )

    def test_empty_window_valid(self) -> None:
        """Пустое окно — корректный дефолт (не ошибка)."""
        window = build_options([])

        assert window.options == ()
        assert window.find("tool:anything") is None


class TestScoreOption:
    """Tests for score_option — totality of the scorer."""

    def test_unknown_tool_degrades_to_epistemic_default(self) -> None:
        """Неизвестная опция → pragmatic=0, epistemic=uncertainty (не падает)."""
        option = Option("tool:brand_new", OptionSource.TOOL)

        pragmatic, epistemic, reason = score_option(
            option, _context(0.7), ActuationPreferences()
        )

        assert pragmatic == 0.0
        assert epistemic == pytest.approx(0.7)
        assert "epistemic" in reason

    def test_relevance_scales_epistemic(self) -> None:
        """relevance ≥ floor — обогащение: uncertainty·relevance."""
        option = Option("tool:theme", OptionSource.TOOL, relevance=0.5)

        _, epistemic, reason = score_option(
            option, _context(0.8), ActuationPreferences()
        )

        assert epistemic == pytest.approx(0.4)
        assert "relevance" in reason

    def test_relevance_below_floor_ignored(self) -> None:
        """relevance < floor — привязка не учитывается (дефолт)."""
        option = Option("tool:off_topic", OptionSource.TOOL, relevance=0.1)
        prefs = ActuationPreferences(relevance_floor=0.3)

        _, epistemic, reason = score_option(option, _context(0.8), prefs)

        assert epistemic == pytest.approx(0.8)
        assert "default" in reason

    def test_builtin_scored_with_audit_reason(self) -> None:
        """BUILTIN тотально оценивается сейчас, reason помечает этап 2."""
        option = Option("builtin:speak", OptionSource.BUILTIN)

        pragmatic, epistemic, reason = score_option(
            option, _context(0.6), ActuationPreferences()
        )

        assert pragmatic == 0.0
        assert epistemic == pytest.approx(0.6)
        assert "stage 2" in reason


class TestSelectOption:
    """Tests for select_option — deterministic argmax with full trace."""

    def test_empty_window_chosen_none(self) -> None:
        """Пустое окно → chosen=None, причина «empty window»."""
        trace = select_option(OptionWindow(), _context())

        assert trace.chosen is None
        assert trace.reason == "empty window"
        assert trace.candidates == ()

    def test_deterministic_same_input_same_trace(self) -> None:
        """Одинаковый вход → одинаковая трасса (инвариант FC/IS)."""
        window = build_options([_affordance("a"), _affordance("b")])
        context = _context(0.9)

        first = select_option(window, context)
        second = select_option(window, context)

        assert first == second

    def test_tie_break_is_window_order(self) -> None:
        """Равные ценности — побеждает более ранняя опция в окне."""
        window = build_options([_affordance("second"), _affordance("first")])

        trace = select_option(window, _context(0.5))

        assert trace.chosen is not None
        assert trace.chosen.id == "tool:second"

    def test_irreversible_down_weighted_but_visible(self) -> None:
        """Необратимая занижена, но присутствует (видимость ≠ авторизация)."""
        window = build_options(
            [_affordance("send_mail", reversible=False), _affordance("web_search")],
        )

        trace = select_option(window, _context(0.8))
        by_id = {c.option.id: c for c in trace.candidates}

        assert len(trace.candidates) == 2  # не скрыта
        assert trace.chosen is not None
        assert trace.chosen.id == "tool:web_search"
        assert by_id["tool:send_mail"].value < by_id["tool:web_search"].value

    def test_irreversible_wins_when_others_absent(self) -> None:
        """Если обратимых нет — необратимая выбирается (гейт не здесь)."""
        window = build_options([_affordance("send_mail", reversible=False)])

        trace = select_option(window, _context(0.8))

        assert trace.chosen is not None
        assert trace.chosen.id == "tool:send_mail"

    def test_cost_down_weights(self) -> None:
        """Стоимость снижает ценность (cost_weight)."""
        cheap = Option("tool:cheap", OptionSource.TOOL, reversible=True, cost=0.0)
        pricey = Option("tool:pricey", OptionSource.TOOL, reversible=True, cost=9.0)
        window = OptionWindow((pricey, cheap))

        trace = select_option(window, _context(0.5))

        assert trace.chosen is not None
        assert trace.chosen.id == "tool:cheap"

    def test_reason_derived_from_winner(self) -> None:
        """reason трассы выводится из оценки победителя (не постфактум)."""
        window = build_options([_affordance("web_search")])

        trace = select_option(window, _context(0.8))

        assert trace.chosen is not None
        assert trace.reason.startswith(f"chose {trace.chosen.id}:")

    def test_relevance_enrichment_changes_choice(self) -> None:
        """Обогащение relevance меняет выбор аддитивно (без кода под тул)."""
        context = _context(0.5)
        # равные ценности (0.5·1.0 == дефолт 0.5) — тай-брейк за ранней опцией
        tied = OptionWindow(
            (
                Option("tool:plain", OptionSource.TOOL, reversible=True),
                Option(
                    "tool:themed", OptionSource.TOOL, reversible=True, relevance=1.0
                ),
            )
        )
        assert select_option(tied, context).chosen is tied.options[0]
        # более высокая релевантность поздней опции побеждает раннюю
        ranked = OptionWindow(
            (
                Option("tool:low", OptionSource.TOOL, reversible=True, relevance=0.2),
                Option("tool:high", OptionSource.TOOL, reversible=True, relevance=1.0),
            )
        )
        assert select_option(ranked, context).chosen is ranked.options[1]


class TestValidation:
    """Tests for input validation — ValueError on bad data."""

    def test_option_empty_id(self) -> None:
        """Пустой id опции → ValueError."""
        with pytest.raises(ValueError, match="id must not be empty"):
            Option("", OptionSource.TOOL)

    def test_option_negative_cost(self) -> None:
        """Отрицательная стоимость → ValueError."""
        with pytest.raises(ValueError, match="cost must be >= 0"):
            Option("tool:x", OptionSource.TOOL, cost=-1.0)

    def test_option_relevance_out_of_range(self) -> None:
        """relevance вне [0, 1] → ValueError."""
        with pytest.raises(ValueError, match="relevance must be in"):
            Option("tool:x", OptionSource.TOOL, relevance=1.5)

    def test_context_uncertainty_out_of_range(self) -> None:
        """uncertainty вне [0, 1] → ValueError."""
        with pytest.raises(ValueError, match="uncertainty must be in"):
            OptionContext(uncertainty=-0.1)

    @pytest.mark.parametrize(
        "field",
        ["pragmatic_weight", "epistemic_weight", "cost_weight", "irreversible_penalty"],
    )
    def test_preferences_negative_weights(self, field: str) -> None:
        """Отрицательные веса → ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            ActuationPreferences(**{field: -0.5})

    def test_preferences_relevance_floor_out_of_range(self) -> None:
        """relevance_floor вне [0, 1] → ValueError."""
        with pytest.raises(ValueError, match="relevance_floor must be in"):
            ActuationPreferences(relevance_floor=2.0)
