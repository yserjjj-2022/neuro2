"""Functional Core — pure sparse factorization (S6).

Independent factors (mode / partner / task), Bayesian update of a single factor
without materialising the joint distribution (manifest §3.Е). Deterministic,
side-effect-free. No LLM (ADR-0007).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .models import Factor, FactorizedState, Vector, _normalize


def posterior(factor: Factor) -> Vector:
    """Байесовское обновление фактора: prior ∘ likelihood → нормировка.

    Args:
        factor: Фактор с приором и правдоподобием.

    Returns:
        Нормированное апостериорное распределение shape=(n_states,).
    """
    return _normalize(np.asarray(factor.prior) * np.asarray(factor.likelihood))


def marginal(state: FactorizedState, name: str) -> Vector:
    """Маргинал фактора по имени.

    Args:
        state: Факторизованное состояние.
        name: Имя фактора.

    Returns:
        Нормированное распределение фактора.

    Raises:
        KeyError: Если фактор с таким именем отсутствует.
    """
    for factor in state.factors:
        if factor.name == name:
            return posterior(factor)
    raise KeyError(f"no factor named {name!r}")


def update_factor(
    state: FactorizedState,
    name: str,
    observation: Vector,
    *,
    learning_rate: float = 1.0,
) -> FactorizedState:
    """Обновить один фактор по наблюдению, остальные не трогать (чистая).

    Новый приор фактора — смесь старого апостериора и наблюдения:
    ``prior' = (1 - lr)·posterior + lr·normalize(observation)``. Остальные
    факторы переносятся без изменений (независимость, манифест §3.Е).

    Args:
        state: Текущее факторизованное состояние.
        name: Имя обновляемого фактора.
        observation: Вектор наблюдения shape=(n_states,) (неотрицательный).
        learning_rate: Скорость обновления ∈ (0, 1].

    Returns:
        Новое FactorizedState.

    Raises:
        KeyError: Если фактор не найден.
        ValueError: Если размерность наблюдения не совпадает / lr вне (0, 1].
    """
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
    target = None
    for factor in state.factors:
        if factor.name == name:
            target = factor
            break
    if target is None:
        raise KeyError(f"no factor named {name!r}")
    obs = np.asarray(observation, dtype=np.float64)
    if obs.shape != target.prior.shape:
        raise ValueError(
            f"observation shape {obs.shape} != factor shape {target.prior.shape}"
        )
    if float(np.min(obs)) < 0.0:
        raise ValueError("observation must be non-negative")
    blended = (1.0 - learning_rate) * posterior(target) + learning_rate * _normalize(
        obs
    )
    updated = replace(target, prior=_normalize(blended))
    factors = tuple(updated if f.name == name else f for f in state.factors)
    return FactorizedState(factors=factors)


def argmax_state(factor: Factor) -> str:
    """Наиболее вероятное состояние фактора (MAP).

    Args:
        factor: Фактор.

    Returns:
        Имя состояния с максимальным апостериором (тай-брейк — порядок states).
    """
    dist = posterior(factor)
    return factor.states[int(np.argmax(dist))]
