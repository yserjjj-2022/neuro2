"""Unit tests for policy Functional Core (compute.py).

Pure, deterministic action selection. Covers goal-directed behaviour (changing
Preferences changes the choice without retraining) and explainability (the
trace's reason is derived from the winning candidate).
"""

from __future__ import annotations

import pytest

from src.core.homeostasis import HomeostasisState, HomeostaticSignal
from src.core.policy.compute import evaluate_candidates, select_action
from src.core.policy.models import (
    Action,
    MacroContext,
    PolicyContext,
    Preferences,
)


def _homeostasis(max_deviation: float = 0.0, severity: float = 0.0) -> HomeostasisState:
    """Собрать снимок гомеостаза с заданным отклонением."""
    return HomeostasisState(
        signals=(HomeostaticSignal("battery", severity, max_deviation, False),),
        max_deviation=max_deviation,
        severity=severity,
        is_critical=False,
    )


def _context(
    *,
    f: float = 0.0,
    stress: float = 0.0,
    has_new_message: bool = False,
    max_deviation: float = 0.0,
    mode: str = "free",
) -> PolicyContext:
    """Собрать контекст policy для теста."""
    return PolicyContext(
        f=f,
        valence=0.0,
        stress=stress,
        task="tone",
        homeostasis=_homeostasis(max_deviation=max_deviation),
        has_new_message=has_new_message,
        mode=mode,
    )


class TestEvaluateCandidates:
    """Tests for evaluate_candidates — scoring all actions."""

    def test_all_actions_present_in_order(self) -> None:
        """Все четыре действия присутствуют в стабильном порядке."""
        candidates = evaluate_candidates(_context(), Preferences())
        assert [c.action for c in candidates] == [
            Action.RESPOND,
            Action.SILENT,
            Action.INITIATIVE,
            Action.IDENTIFY_PARTNER,
        ]

    def test_purity(self) -> None:
        """Одинаковый вход → одинаковый выход."""
        ctx = _context(has_new_message=True)
        prefs = Preferences()
        assert evaluate_candidates(ctx, prefs) == evaluate_candidates(ctx, prefs)

    def test_respond_scored_on_new_message(self) -> None:
        """Новое сообщение → RESPOND имеет высокую прагматическую ценность."""
        candidates = {
            c.action: c
            for c in evaluate_candidates(_context(has_new_message=True), Preferences())
        }
        assert candidates[Action.RESPOND].pragmatic == 1.0

    def test_respond_zero_without_message(self) -> None:
        """Без сообщения RESPOND не получает ценности."""
        candidates = {
            c.action: c for c in evaluate_candidates(_context(), Preferences())
        }
        assert candidates[Action.RESPOND].pragmatic == 0.0

    def test_identify_partner_disabled(self) -> None:
        """IDENTIFY_PARTNER — заготовка (нулевая ценность на S4)."""
        candidates = {
            c.action: c for c in evaluate_candidates(_context(), Preferences())
        }
        assert candidates[Action.IDENTIFY_PARTNER].value == 0.0


class TestSelectAction:
    """Tests for select_action — deterministic choice + trace."""

    def test_respond_on_new_message(self) -> None:
        """Новое сообщение → RESPOND."""
        trace = select_action(_context(has_new_message=True), Preferences())
        assert trace.chosen is Action.RESPOND

    def test_silent_when_nothing_happens(self) -> None:
        """Тихий тик без триггеров → SILENT (безопасный дефолт)."""
        trace = select_action(_context(), Preferences())
        assert trace.chosen is Action.SILENT

    def test_initiative_on_high_f(self) -> None:
        """F выше порога → INITIATIVE."""
        trace = select_action(_context(f=5.0), Preferences())
        assert trace.chosen is Action.INITIATIVE

    def test_initiative_on_homeostatic_alert(self) -> None:
        """Высокое отклонение гомеостаза → INITIATIVE (предупредить)."""
        trace = select_action(_context(max_deviation=0.9), Preferences())
        assert trace.chosen is Action.INITIATIVE

    def test_deterministic(self) -> None:
        """Одинаковый вход → одинаковая трасса."""
        ctx = _context(has_new_message=True)
        prefs = Preferences()
        assert select_action(ctx, prefs) == select_action(ctx, prefs)

    def test_trace_contains_all_candidates(self) -> None:
        """Трасса содержит всех кандидатов с оценками."""
        trace = select_action(_context(has_new_message=True), Preferences())
        assert len(trace.candidates) == 4
        assert trace.chosen in {c.action for c in trace.candidates}

    def test_reason_derived_from_winner(self) -> None:
        """Причина выбора соответствует действию победителя (explainability)."""
        trace = select_action(_context(has_new_message=True), Preferences())
        assert "respond" in trace.reason
        assert trace.reason != ""


class TestGoalDirected:
    """Goal-directed test: смена Preferences меняет выбор без переобучения."""

    def test_disable_responding(self) -> None:
        """respond_to_messages=False → на сообщение не отвечаем (SILENT)."""
        ctx = _context(has_new_message=True)
        assert select_action(ctx, Preferences()).chosen is Action.RESPOND
        assert (
            select_action(ctx, Preferences(respond_to_messages=False)).chosen
            is Action.SILENT
        )

    def test_disable_homeostatic_alert(self) -> None:
        """homeostatic_alert=False → тревога не вызывает инициативу."""
        ctx = _context(max_deviation=0.9)
        assert select_action(ctx, Preferences()).chosen is Action.INITIATIVE
        assert (
            select_action(ctx, Preferences(homeostatic_alert=False)).chosen
            is Action.SILENT
        )

    def test_change_initiative_threshold(self) -> None:
        """Порог инициативы управляет её срабатыванием."""
        ctx = _context(f=2.0)
        assert select_action(ctx, Preferences()).chosen is Action.INITIATIVE
        assert (
            select_action(ctx, Preferences(initiative_f_threshold=10.0)).chosen
            is Action.SILENT
        )


class TestPreferencesValidation:
    """Tests for Preferences __post_init__ validation."""

    def test_negative_threshold_raises(self) -> None:
        """Отрицательный порог → ValueError."""
        with pytest.raises(ValueError):
            Preferences(initiative_f_threshold=-1.0)

    def test_negative_weight_raises(self) -> None:
        """Отрицательный вес → ValueError."""
        with pytest.raises(ValueError):
            Preferences(pragmatic_weight=-1.0)
        with pytest.raises(ValueError):
            Preferences(epistemic_weight=-1.0)

    def test_alert_deviation_out_of_range_raises(self) -> None:
        """alert_deviation вне [0, 1] → ValueError."""
        with pytest.raises(ValueError):
            Preferences(alert_deviation=-0.1)
        with pytest.raises(ValueError):
            Preferences(alert_deviation=1.1)

    def test_negative_silent_params_raise(self) -> None:
        """silent_baseline/silent_stress_gain < 0 → ValueError."""
        with pytest.raises(ValueError):
            Preferences(silent_baseline=-1.0)
        with pytest.raises(ValueError):
            Preferences(silent_stress_gain=-1.0)

    def test_configurable_alert_deviation(self) -> None:
        """alert_deviation из Preferences управляет срабатыванием тревоги."""
        ctx = _context(max_deviation=0.5)
        assert (
            select_action(ctx, Preferences(alert_deviation=0.4)).chosen
            is Action.INITIATIVE
        )
        assert (
            select_action(ctx, Preferences(alert_deviation=0.8)).chosen is Action.SILENT
        )


class TestMacroContext:
    """Tests for MacroContext — minimal discrete layer (S4)."""

    def test_defaults(self) -> None:
        """Дефолты: task=none, mode=free."""
        macro = MacroContext()
        assert macro.task == "none"
        assert macro.mode == "free"

    def test_valid_modes(self) -> None:
        """Все три режима валидны."""
        for mode in ("game", "cooperative", "free"):
            assert MacroContext(mode=mode).mode == mode

    def test_invalid_mode_raises(self) -> None:
        """Неизвестный режим → ValueError."""
        with pytest.raises(ValueError):
            MacroContext(mode="sandbox")

    def test_used_by_policy_context(self) -> None:
        """MacroContext переносится в PolicyContext (task/mode)."""
        macro = MacroContext(task="tone", mode="game")
        ctx = PolicyContext(
            f=0.0,
            valence=0.0,
            stress=0.0,
            task=macro.task,
            homeostasis=_homeostasis(),
            has_new_message=False,
            mode=macro.mode,
        )
        assert ctx.task == "tone"
        assert ctx.mode == "game"
        assert (
            select_action(ctx, Preferences(alert_deviation=0.8)).chosen is Action.SILENT
        )
