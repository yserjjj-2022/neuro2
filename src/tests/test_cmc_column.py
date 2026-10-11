"""Unit tests for CMC Functional Core — column_step and ColumnConfig.

Core is pure: tested without ensemble, without state, without I/O.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.cmc.column import column_step
from src.core.cmc.models import ColumnConfig, ColumnState


@pytest.fixture()
def cfg() -> ColumnConfig:
    return ColumnConfig(input_dim=3, state_dim=3)


@pytest.fixture()
def zero_state(cfg: ColumnConfig) -> ColumnState:
    return ColumnState(x=np.zeros(cfg.input_dim), e=np.zeros(cfg.input_dim))


def test_config_invalid_dims() -> None:
    """input_dim <= 0 → ValueError (fail-fast в конфиге)."""
    with pytest.raises(ValueError):
        ColumnConfig(input_dim=0, state_dim=3)
    with pytest.raises(ValueError):
        ColumnConfig(input_dim=3, state_dim=-1)


def test_config_invalid_alpha() -> None:
    """alpha ∉ [0, 1] → ValueError, не клиппинг."""
    with pytest.raises(ValueError):
        ColumnConfig(input_dim=3, state_dim=3, alpha=-0.1)
    with pytest.raises(ValueError):
        ColumnConfig(input_dim=3, state_dim=3, alpha=1.5)


def test_config_state_neq_input() -> None:
    """state_dim != input_dim → ValueError (в Фазе 1 нет матриц проекций)."""
    with pytest.raises(ValueError):
        ColumnConfig(input_dim=3, state_dim=5)


def test_column_step_first(cfg: ColumnConfig, zero_state: ColumnState) -> None:
    """Первый шаг из нулевого состояния: e(0) = u(0), x(0) = α·u(0)."""
    u = np.array([1.0, 2.0, 3.0])

    result = column_step(cfg, u, zero_state)

    np.testing.assert_allclose(result.x, cfg.alpha * u)
    np.testing.assert_allclose(result.e, u - cfg.alpha * u)


def test_column_step_purity(cfg: ColumnConfig, zero_state: ColumnState) -> None:
    """Чистота: одинаковый вход → одинаковый выход; вход не мутируется."""
    u = np.array([1.0, 2.0, 3.0])

    r1 = column_step(cfg, u, zero_state)
    r2 = column_step(cfg, u, zero_state)

    np.testing.assert_array_equal(r1.x, r2.x)
    np.testing.assert_array_equal(r1.e, r2.e)

    # Мутация u после вызова не влияет на результат
    u[:] = 999.0
    np.testing.assert_array_equal(r1.x, cfg.alpha * np.array([1.0, 2.0, 3.0]))


def test_column_step_convergence(cfg: ColumnConfig, zero_state: ColumnState) -> None:
    """Стабильный вход: e(t) → 0 (геометрическая сходимость)."""
    u = np.array([1.0, 2.0, 3.0])
    state = zero_state

    for _ in range(200):
        state = column_step(cfg, u, state)

    np.testing.assert_allclose(state.e, np.zeros(cfg.input_dim), atol=1e-8)


def test_column_step_shape_mismatch(cfg: ColumnConfig, zero_state: ColumnState) -> None:
    """u.shape != prev.x.shape → ValueError (fail-fast)."""
    u = np.array([1.0, 2.0])  # (2,) vs (3,)

    with pytest.raises(ValueError):
        column_step(cfg, u, zero_state)


def test_column_step_mask_shape_mismatch(
    cfg: ColumnConfig, zero_state: ColumnState
) -> None:
    """mask.shape != u.shape → ValueError (fail-fast)."""
    u = np.array([1.0, 2.0, 3.0])
    mask = np.array([1.0, 0.0])  # (2,) vs (3,)

    with pytest.raises(ValueError):
        column_step(cfg, u, zero_state, mask)


def test_column_step_mask_is_not_observed(
    cfg: ColumnConfig, zero_state: ColumnState
) -> None:
    """Маска «не наблюдается»: нечитаемые координаты → x=0, e=0 (ADR-0013 §2)."""
    u = np.array([1.0, 2.0, 3.0])
    mask = np.array([1.0, 0.0, 0.0])

    result = column_step(cfg, u, zero_state, mask)

    # Читаемая координата 0 обновляется как обычно; 1,2 — не наблюдается.
    np.testing.assert_allclose(result.x, [cfg.alpha * 1.0, 0.0, 0.0])
    np.testing.assert_allclose(result.e, [1.0 - cfg.alpha * 1.0, 0.0, 0.0])


def test_column_step_mask_no_phantom_error(
    cfg: ColumnConfig, zero_state: ColumnState
) -> None:
    """Ноль на отключённой координате не создаёт ошибку (не «наблюдение 0»)."""
    # u=0 на координатах 1,2 — но они не читаются, ошибки быть не должно.
    u = np.array([5.0, 0.0, 0.0])
    mask = np.array([1.0, 0.0, 0.0])

    result = column_step(cfg, u, zero_state, mask)

    assert result.e[1] == 0.0
    assert result.e[2] == 0.0


def test_column_step_mask_convergence_read_only(
    cfg: ColumnConfig, zero_state: ColumnState
) -> None:
    """Сходимость только на читаемых координатах; нечитаемые — ровно 0."""
    u = np.array([1.0, 2.0, 3.0])
    mask = np.array([1.0, 0.0, 0.0])
    state = zero_state

    for _ in range(200):
        state = column_step(cfg, u, state, mask)

    np.testing.assert_allclose(state.e, np.zeros(3), atol=1e-8)


def test_column_step_mask_equals_manual_legacy(
    cfg: ColumnConfig, zero_state: ColumnState
) -> None:
    """Маска «все единицы» тождественна mask=None (legacy-инвариант)."""
    u = np.array([1.0, 2.0, 3.0])
    ones = np.ones(3)

    with_mask = column_step(cfg, u, zero_state, ones)
    without = column_step(cfg, u, zero_state, None)

    np.testing.assert_array_equal(with_mask.x, without.x)
    np.testing.assert_array_equal(with_mask.e, without.e)


def test_column_step_mask_not_mutated(
    cfg: ColumnConfig, zero_state: ColumnState
) -> None:
    """Чистота: mask не мутируется вызовом."""
    u = np.array([1.0, 2.0, 3.0])
    mask = np.array([1.0, 0.0, 0.0])
    before = mask.copy()

    column_step(cfg, u, zero_state, mask)

    np.testing.assert_array_equal(mask, before)
