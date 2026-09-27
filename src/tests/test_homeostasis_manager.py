"""Unit tests for Homeostat Shell (manager.py).

Homeostat reads raw SignalSource severities from the sensory bus and compares
them to setpoints. Channels without a setpoint are ignored; order follows the
setpoint order.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.homeostasis.manager import Homeostat
from src.core.homeostasis.models import Setpoint
from src.mcp import SignalCategory, SignalSource


def _intero(tag: str, severity: float, dim: int = 1) -> SignalSource:
    """Собрать интероцептивный сигнал заданной severity."""
    return SignalSource(
        category=SignalCategory.INTEROCEPTIVE,
        data=np.zeros(dim, dtype=np.float64),
        severity=severity,
        tag=tag,
    )


class TestSetpointValidation:
    """Tests for Setpoint __post_init__ validation."""

    def test_defaults(self) -> None:
        """Дефолтный сетпоинт валиден."""
        sp = Setpoint(tag="battery")
        assert sp.comfort == 0.5
        assert sp.critical == 0.9
        assert sp.weight == 1.0

    def test_empty_tag_raises(self) -> None:
        """Пустой tag → ValueError."""
        with pytest.raises(ValueError):
            Setpoint(tag="")

    def test_comfort_out_of_range_raises(self) -> None:
        """comfort вне [0, 1] → ValueError."""
        with pytest.raises(ValueError):
            Setpoint(tag="battery", comfort=-0.1)
        with pytest.raises(ValueError):
            Setpoint(tag="battery", comfort=1.1)

    def test_critical_out_of_range_raises(self) -> None:
        """critical вне [0, 1] → ValueError."""
        with pytest.raises(ValueError):
            Setpoint(tag="battery", critical=1.1)

    def test_comfort_not_below_critical_raises(self) -> None:
        """comfort >= critical → ValueError."""
        with pytest.raises(ValueError):
            Setpoint(tag="battery", comfort=0.9, critical=0.9)

    def test_nonpositive_weight_raises(self) -> None:
        """weight <= 0 → ValueError."""
        with pytest.raises(ValueError):
            Setpoint(tag="battery", weight=0.0)


class TestHomeostatInit:
    """Tests for Homeostat __init__ validation."""

    def test_empty_setpoints_raises(self) -> None:
        """Пустой список сетпоинтов → ValueError."""
        with pytest.raises(ValueError):
            Homeostat(setpoints=[])

    def test_reflex_threshold_out_of_range_raises(self) -> None:
        """reflex_threshold вне [0, 1] → ValueError."""
        with pytest.raises(ValueError):
            Homeostat(setpoints=[Setpoint(tag="battery")], reflex_threshold=1.5)


class TestHomeostatEvaluate:
    """Tests for Homeostat.evaluate — bus signals → HomeostasisState."""

    def test_no_signals_is_norm(self) -> None:
        """Нет сигналов → все каналы в норме (severity 0)."""
        h = Homeostat(setpoints=[Setpoint(tag="battery")])
        state = h.evaluate([])
        assert state.max_deviation == 0.0
        assert state.severity == 0.0
        assert state.is_critical is False
        assert state.signals[0].tag == "battery"

    def test_nominal_channel(self) -> None:
        """severity ниже comfort → deviation 0, не критично."""
        h = Homeostat(setpoints=[Setpoint(tag="battery")])
        state = h.evaluate([_intero("battery", 0.2)])
        assert state.max_deviation == 0.0
        assert state.is_critical is False

    def test_critical_channel(self) -> None:
        """severity >= 0.9 → is_critical True, deviation 1.0."""
        h = Homeostat(setpoints=[Setpoint(tag="battery")])
        state = h.evaluate([_intero("battery", 0.95)])
        assert state.is_critical is True
        assert state.max_deviation == 1.0
        assert state.severity == pytest.approx(0.95)

    def test_unknown_channel_ignored(self) -> None:
        """Канал без сетпоинта игнорируется."""
        h = Homeostat(setpoints=[Setpoint(tag="battery")])
        state = h.evaluate([_intero("battery", 0.2), _intero("cpu", 1.0)])
        assert len(state.signals) == 1
        assert state.signals[0].tag == "battery"
        assert state.is_critical is False

    def test_order_follows_setpoints(self) -> None:
        """Порядок результата = порядок сетпоинтов (детерминизм)."""
        h = Homeostat(setpoints=[Setpoint(tag="resources"), Setpoint(tag="battery")])
        state = h.evaluate([_intero("battery", 0.9), _intero("resources", 0.6)])
        assert [s.tag for s in state.signals] == ["resources", "battery"]

    def test_max_deviation_over_channels(self) -> None:
        """max_deviation — максимум по каналам."""
        h = Homeostat(setpoints=[Setpoint(tag="battery"), Setpoint(tag="resources")])
        state = h.evaluate([_intero("battery", 0.5), _intero("resources", 0.9)])
        assert state.max_deviation == 1.0
        assert state.severity == pytest.approx(0.9)
        assert state.is_critical is True

    def test_determinism(self) -> None:
        """Одинаковый вход → одинаковый снимок."""
        h = Homeostat(setpoints=[Setpoint(tag="battery")])
        signals = [_intero("battery", 0.7)]
        assert h.evaluate(signals) == h.evaluate(signals)

    def test_custom_reflex_threshold(self) -> None:
        """Кастомный порог рефлекса учитывается в is_critical."""
        h = Homeostat(setpoints=[Setpoint(tag="battery")], reflex_threshold=0.7)
        state = h.evaluate([_intero("battery", 0.75)])
        assert state.is_critical is True
