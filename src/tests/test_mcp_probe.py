"""Unit tests for MCP probing contract (S6 проход 2): affordances + selection."""

from __future__ import annotations

import pytest

from src.mcp import (
    Affordance,
    AffordanceMap,
    SignalCategory,
    default_affordances,
    select_affordance,
)


class TestAffordance:
    def test_validation_name_and_dim(self) -> None:
        with pytest.raises(ValueError):
            Affordance("", SignalCategory.EXTEROCEPTIVE)
        with pytest.raises(ValueError):
            Affordance("x", SignalCategory.EXTEROCEPTIVE, dim=0)


class TestAffordanceMap:
    def test_duplicate_rejected(self) -> None:
        a = Affordance("x", SignalCategory.EXTEROCEPTIVE)
        with pytest.raises(ValueError):
            AffordanceMap((a, a))

    def test_find_and_names(self) -> None:
        amap = default_affordances()
        assert amap.find("web_search") is not None
        assert amap.find("missing") is None
        assert "web_search" in amap.names

    def test_reversible_filter(self) -> None:
        amap = AffordanceMap(
            (
                Affordance("a", SignalCategory.EXTEROCEPTIVE, reversible=True),
                Affordance("b", SignalCategory.EXTEROCEPTIVE, reversible=False),
            )
        )
        assert [a.name for a in amap.reversible] == ["a"]


class TestSelectAffordance:
    def test_below_threshold_none(self) -> None:
        assert select_affordance(0.1, default_affordances(), threshold=0.5) is None

    def test_above_threshold_picks_first_reversible(self) -> None:
        chosen = select_affordance(0.9, default_affordances(), threshold=0.5)
        assert chosen is not None
        assert chosen.name == "web_search"

    def test_no_reversible_none(self) -> None:
        amap = AffordanceMap(
            (Affordance("a", SignalCategory.EXTEROCEPTIVE, reversible=False),)
        )
        assert select_affordance(0.9, amap, threshold=0.5) is None

    def test_threshold_validation(self) -> None:
        with pytest.raises(ValueError):
            select_affordance(0.5, default_affordances(), threshold=2.0)
