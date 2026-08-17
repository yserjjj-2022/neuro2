"""Unit tests for MCP signal models.

Frozen dataclass + enum — аналог FreeEnergyResult (energy), VotingResult (voting).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.mcp.models import SignalCategory, SignalSource, Vector


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


def test_signal_category_enum() -> None:
    """Все три категории существуют и имеют правильные значения."""
    assert SignalCategory.EXTEROCEPTIVE.value == "exteroceptive"
    assert SignalCategory.INTEROCEPTIVE.value == "interoceptive"
    assert SignalCategory.COMMUNICATIVE.value == "communicative"


def test_signal_source_frozen() -> None:
    """SignalSource — frozen dataclass, мутация запрещена."""
    sig = SignalSource(
        category=SignalCategory.EXTEROCEPTIVE,
        data=np.array([1.0], dtype=np.float64),
    )
    with pytest.raises(Exception):
        sig.severity = 0.5  # type: ignore[assignment]


def test_signal_source_valid() -> None:
    """Валидный сигнал создаётся без исключений."""
    sig = SignalSource(
        category=SignalCategory.EXTEROCEPTIVE,
        data=np.array([1.0, 2.0], dtype=np.float64),
        severity=0.3,
        tag="test",
    )
    assert sig.category == SignalCategory.EXTEROCEPTIVE
    assert sig.severity == 0.3
    assert sig.is_reflex is False
    assert sig.tag == "test"


def test_signal_source_severity_too_low() -> None:
    """severity < 0.0 → ValueError."""
    with pytest.raises(ValueError, match="severity must be in \\[0.0, 1.0\\]"):
        SignalSource(
            category=SignalCategory.EXTEROCEPTIVE,
            data=np.array([1.0], dtype=np.float64),
            severity=-0.1,
        )


def test_signal_source_severity_too_high() -> None:
    """severity > 1.0 → ValueError."""
    with pytest.raises(ValueError, match="severity must be in \\[0.0, 1.0\\]"):
        SignalSource(
            category=SignalCategory.EXTEROCEPTIVE,
            data=np.array([1.0], dtype=np.float64),
            severity=1.1,
        )


def test_signal_source_reflex_auto_set() -> None:
    """interoceptive severity ≥ 0.9 → is_reflex=True."""
    sig = SignalSource(
        category=SignalCategory.INTEROCEPTIVE,
        data=np.array([0.1], dtype=np.float64),
        severity=0.95,
        tag="battery_critical",
    )
    assert sig.is_reflex is True


def test_signal_source_reflex_non_interoceptive() -> None:
    """exteroceptive is_reflex=True → ValueError."""
    with pytest.raises(ValueError, match="is_reflex=True only allowed"):
        SignalSource(
            category=SignalCategory.EXTEROCEPTIVE,
            data=np.array([1.0], dtype=np.float64),
            severity=0.5,
            is_reflex=True,
            tag="weather",
        )


def test_signal_source_communicative() -> None:
    """Communicative сигнал без reflex."""
    sig = SignalSource(
        category=SignalCategory.COMMUNICATIVE,
        data=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        severity=0.1,
        tag="user_message",
    )
    assert sig.category == SignalCategory.COMMUNICATIVE
    assert sig.is_reflex is False
    assert sig.tag == "user_message"


def test_signal_source_data_not_mutated() -> None:
    """Входной массив data не мутируется при создании."""
    original = np.array([1.0, 2.0], dtype=np.float64)
    sig = SignalSource(
        category=SignalCategory.EXTEROCEPTIVE,
        data=original.copy(),
    )
    # frozen dataclass автоматически хранит копию через dataclass
    # но для numpy это не гарантировано — проверяем что original не изменился
    np.testing.assert_array_equal(sig.data, original)
