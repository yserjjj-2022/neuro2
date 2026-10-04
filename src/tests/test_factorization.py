"""Unit tests for sparse factorization Core (S6)."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.factorization import (
    Factor,
    FactorizedState,
    argmax_state,
    marginal,
    posterior,
    update_factor,
)


def _factor(
    name: str = "mode",
    states: tuple[str, ...] = ("game", "cooperative", "free"),
    prior: tuple[float, ...] = (0.2, 0.3, 0.5),
    likelihood: tuple[float, ...] = (0.5, 0.3, 0.2),
) -> Factor:
    return Factor(
        name=name,
        states=states,
        prior=np.array(prior),
        likelihood=np.array(likelihood),
    )


class TestFactorValidation:
    def test_empty_states(self) -> None:
        with pytest.raises(ValueError):
            Factor(name="x", states=(), prior=np.array([]), likelihood=np.array([]))

    def test_duplicate_states(self) -> None:
        with pytest.raises(ValueError):
            _factor(states=("a", "a", "b"))

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            Factor(
                name="x",
                states=("a", "b"),
                prior=np.array([1.0, 0.0, 0.0]),
                likelihood=np.array([1.0, 0.0]),
            )

    def test_negative_values(self) -> None:
        with pytest.raises(ValueError):
            Factor(
                name="x",
                states=("a", "b"),
                prior=np.array([1.0, -1.0]),
                likelihood=np.array([1.0, 1.0]),
            )


class TestPosterior:
    def test_normalized(self) -> None:
        dist = posterior(_factor())
        assert dist.shape == (3,)
        assert float(np.sum(dist)) == pytest.approx(1.0)

    def test_zero_likelihood_uniform(self) -> None:
        f = _factor(prior=(0.0, 0.0, 0.0), likelihood=(0.0, 0.0, 0.0))
        dist = posterior(f)
        assert np.allclose(dist, np.full(3, 1 / 3))

    def test_shifts_toward_likelihood(self) -> None:
        f = _factor(prior=(0.5, 0.5), states=("a", "b"), likelihood=(0.9, 0.1))
        dist = posterior(f)
        assert dist[0] > dist[1]


class TestMarginalAndState:
    def test_marginal_by_name(self) -> None:
        state = FactorizedState(
            factors=(
                _factor("mode"),
                _factor("partner", states=("unknown", "known"),
                        prior=(0.5, 0.5), likelihood=(0.6, 0.4)),
            )
        )
        assert marginal(state, "partner").shape == (2,)

    def test_marginal_missing_raises(self) -> None:
        state = FactorizedState(factors=(_factor(),))
        with pytest.raises(KeyError):
            marginal(state, "nope")

    def test_argmax(self) -> None:
        f = _factor(prior=(0.1, 0.1, 0.8), likelihood=(1.0, 1.0, 1.0))
        assert argmax_state(f) == "free"

    def test_duplicate_names_raise(self) -> None:
        with pytest.raises(ValueError):
            FactorizedState(factors=(_factor("m"), _factor("m")))

    def test_empty_factors_raise(self) -> None:
        with pytest.raises(ValueError):
            FactorizedState(factors=())


class TestUpdateFactor:
    def test_isolated_update(self) -> None:
        state = FactorizedState(
            factors=(
                _factor("mode"),
                _factor("partner", states=("unknown", "known"),
                        prior=(0.5, 0.5), likelihood=(0.6, 0.4)),
            )
        )
        before_partner = marginal(state, "partner")
        updated = update_factor(state, "mode", np.array([1.0, 0.0, 0.0]))
        # partner не изменился
        assert np.allclose(marginal(updated, "partner"), before_partner)
        # mode сдвинулся к первому состоянию
        assert marginal(updated, "mode")[0] > marginal(state, "mode")[0]

    def test_learning_rate_validation(self) -> None:
        state = FactorizedState(factors=(_factor(),))
        with pytest.raises(ValueError):
            update_factor(state, "mode", np.ones(3), learning_rate=0.0)
        with pytest.raises(ValueError):
            update_factor(state, "mode", np.ones(3), learning_rate=1.5)

    def test_observation_shape(self) -> None:
        state = FactorizedState(factors=(_factor(),))
        with pytest.raises(ValueError):
            update_factor(state, "mode", np.ones(5))

    def test_negative_observation(self) -> None:
        state = FactorizedState(factors=(_factor(),))
        with pytest.raises(ValueError):
            update_factor(state, "mode", np.array([1.0, -1.0, 0.0]))

    def test_missing_factor_raises(self) -> None:
        state = FactorizedState(factors=(_factor(),))
        with pytest.raises(KeyError):
            update_factor(state, "nope", np.ones(3))

    def test_deterministic(self) -> None:
        state = FactorizedState(factors=(_factor(),))
        obs = np.array([0.5, 0.3, 0.2])
        a = update_factor(state, "mode", obs)
        b = update_factor(state, "mode", obs)
        assert np.allclose(marginal(a, "mode"), marginal(b, "mode"))

    def test_immutability(self) -> None:
        state = FactorizedState(factors=(_factor(),))
        before = marginal(state, "mode").copy()
        update_factor(state, "mode", np.array([1.0, 0.0, 0.0]))
        assert np.allclose(marginal(state, "mode"), before)
