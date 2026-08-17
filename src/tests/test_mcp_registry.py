"""Unit tests for SignalRegistry — Imperative Shell.

Registry owns sources list, tested with in-memory registry.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.mcp.models import SignalCategory, SignalSource
from src.mcp.registry import SignalRegistry


@pytest.fixture()
def registry() -> SignalRegistry:
    return SignalRegistry(active_threshold=0.0)


@pytest.fixture()
def extero_signal() -> SignalSource:
    return SignalSource(
        category=SignalCategory.EXTEROCEPTIVE,
        data=np.array([1.0, 2.0], dtype=np.float64),
        severity=0.3,
        tag="weather",
    )


@pytest.fixture()
def intero_signal() -> SignalSource:
    return SignalSource(
        category=SignalCategory.INTEROCEPTIVE,
        data=np.array([0.85], dtype=np.float64),
        severity=0.85,
        tag="battery",
    )


@pytest.fixture()
def reflex_signal() -> SignalSource:
    return SignalSource(
        category=SignalCategory.INTEROCEPTIVE,
        data=np.array([0.1], dtype=np.float64),
        severity=0.95,
        tag="battery_critical",
    )


def test_registry_empty(registry: SignalRegistry) -> None:
    """Пустой registry → aggregate() = None."""
    assert registry.aggregate() is None
    assert registry.count == 0


def test_registry_register(
    registry: SignalRegistry, extero_signal: SignalSource
) -> None:
    """register → count == 1."""
    registry.register(extero_signal)
    assert registry.count == 1
    assert len(registry.sources) == 1


def test_registry_unregister_by_tag(
    registry: SignalRegistry, extero_signal: SignalSource
) -> None:
    """Unregister существующего tag → count == 0, возвращает True."""
    registry.register(extero_signal)
    result = registry.unregister_by_tag("weather")
    assert result is True
    assert registry.count == 0


def test_registry_unregister_nonexistent(registry: SignalRegistry) -> None:
    """Unregister несуществующего tag → возвращает False."""
    result = registry.unregister_by_tag("nonexistent")
    assert result is False
    assert registry.count == 0


def test_registry_clear(registry: SignalRegistry) -> None:
    """Clear → count == 0."""
    registry.register(
        SignalSource(
            category=SignalCategory.EXTEROCEPTIVE,
            data=np.array([1.0], dtype=np.float64),
            tag="a",
        )
    )
    registry.register(
        SignalSource(
            category=SignalCategory.INTEROCEPTIVE,
            data=np.array([0.5], dtype=np.float64),
            tag="b",
        )
    )
    assert registry.count == 2
    registry.clear()
    assert registry.count == 0


def test_registry_get_by_category(
    registry: SignalRegistry,
    extero_signal: SignalSource,
    intero_signal: SignalSource,
) -> None:
    """Фильтрация по категории."""
    registry.register(extero_signal)
    registry.register(intero_signal)

    extero = registry.get_by_category(SignalCategory.EXTEROCEPTIVE)
    intero = registry.get_by_category(SignalCategory.INTEROCEPTIVE)

    assert len(extero) == 1
    assert extero[0].tag == "weather"
    assert len(intero) == 1
    assert intero[0].tag == "battery"


def test_registry_get_reflex_signals(
    registry: SignalRegistry,
    intero_signal: SignalSource,
    reflex_signal: SignalSource,
) -> None:
    """Reflex-фильтрация: только severity ≥ 0.9."""
    registry.register(intero_signal)
    registry.register(reflex_signal)

    reflex = registry.get_reflex_signals()
    assert len(reflex) == 1
    assert reflex[0].tag == "battery_critical"


def test_registry_aggregate(registry: SignalRegistry) -> None:
    """Агрегация одного сигнала."""
    sig = SignalSource(
        category=SignalCategory.EXTEROCEPTIVE,
        data=np.array([1.0, 2.0], dtype=np.float64),
        tag="test",
    )
    registry.register(sig)

    result = registry.aggregate()
    assert result is not None
    np.testing.assert_array_equal(result, np.array([1.0, 2.0], dtype=np.float64))


def test_registry_aggregate_multiple(registry: SignalRegistry) -> None:
    """Агрегация нескольких сигналов → суммарный shape."""
    sig1 = SignalSource(
        category=SignalCategory.EXTEROCEPTIVE,
        data=np.array([1.0, 2.0], dtype=np.float64),
        tag="a",
    )
    sig2 = SignalSource(
        category=SignalCategory.INTEROCEPTIVE,
        data=np.array([0.5], dtype=np.float64),
        tag="b",
    )
    registry.register(sig1)
    registry.register(sig2)

    result = registry.aggregate()
    assert result is not None
    assert result.shape == (3,)
    np.testing.assert_array_equal(
        result, np.array([1.0, 2.0, 0.5], dtype=np.float64)
    )


def test_registry_aggregate_with_reflex(
    registry: SignalRegistry,
    intero_signal: SignalSource,
    reflex_signal: SignalSource,
) -> None:
    """Агрегация включает reflex-сигналы."""
    registry.register(intero_signal)
    registry.register(reflex_signal)

    result = registry.aggregate()
    assert result is not None
    assert result.shape == (2,)


def test_registry_sources_readonly(registry: SignalRegistry) -> None:
    """sources — read-only view (копия, не ссылка)."""
    sig = SignalSource(
        category=SignalCategory.EXTEROCEPTIVE,
        data=np.array([1.0], dtype=np.float64),
        tag="test",
    )
    registry.register(sig)

    srcs = registry.sources
    srcs.clear()  # мутирует копию
    assert registry.count == 1  # registry не изменился
