"""Unit tests for VotingManager — Imperative Shell.

Shell owns k and caches the last result; delegates to pure kwta().
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.voting.manager import VotingManager
from src.core.voting.models import VotingResult


def test_manager_default_k() -> None:
    """k=1 по умолчанию (hard-WTA)."""
    manager = VotingManager()
    assert manager.last is None

    result = manager.vote(np.array([0.3, 0.8, 0.1]))
    assert len(result.indices) == 1


def test_manager_invalid_k_init() -> None:
    """k < 1 в конструкторе → ValueError."""
    with pytest.raises(ValueError):
        VotingManager(k=0)
    with pytest.raises(ValueError):
        VotingManager(k=-5)


def test_manager_vote() -> None:
    """vote() возвращает VotingResult с корректными индексами."""
    manager = VotingManager(k=2)
    scores = np.array([0.1, 0.9, 0.4, 0.7])

    result = manager.vote(scores)

    assert isinstance(result, VotingResult)
    assert result.indices.tolist() == [1, 3]


def test_manager_vote_k_gt_n() -> None:
    """k > N → ValueError (делегируется в kwta)."""
    manager = VotingManager(k=5)
    scores = np.array([0.1, 0.2])

    with pytest.raises(ValueError):
        manager.vote(scores)


def test_manager_last_none_before() -> None:
    """last is None до первого vote()."""
    manager = VotingManager()
    assert manager.last is None


def test_manager_last_after() -> None:
    """last = результат после vote()."""
    manager = VotingManager(k=1)
    scores = np.array([0.2, 0.8])

    result = manager.vote(scores)

    assert manager.last is result
    assert result.indices[0] == 1


def test_manager_set_k() -> None:
    """set_k меняет число победителей; k < 1 → ValueError."""
    manager = VotingManager(k=1)
    scores = np.array([0.1, 0.6, 0.4, 0.9])

    result = manager.vote(scores)
    assert len(result.indices) == 1

    manager.set_k(2)
    result = manager.vote(scores)
    assert len(result.indices) == 2

    with pytest.raises(ValueError):
        manager.set_k(0)
