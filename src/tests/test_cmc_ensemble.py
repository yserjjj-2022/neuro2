"""Unit tests for CMCEnsemble — Imperative Shell.

Shell owns column states; tested with small in-memory ensembles.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.cmc.ensemble import CMCEnsemble
from src.core.cmc.models import ColumnConfig


@pytest.fixture()
def ensemble() -> CMCEnsemble:
    return CMCEnsemble(
        columns=[
            ColumnConfig(input_dim=3, state_dim=3, specialization="tone"),
            ColumnConfig(input_dim=3, state_dim=3, specialization="rhythm"),
            ColumnConfig(input_dim=3, state_dim=3, specialization="meaning"),
        ]
    )


def test_ensemble_init_empty() -> None:
    """Пустой список колонок → ValueError."""
    with pytest.raises(ValueError):
        CMCEnsemble(columns=[])


def test_ensemble_init_mismatched_dims() -> None:
    """Разные input_dim/state_dim у колонок → ValueError."""
    with pytest.raises(ValueError):
        CMCEnsemble(
            columns=[
                ColumnConfig(input_dim=3, state_dim=3),
                ColumnConfig(input_dim=5, state_dim=5),
            ]
        )


def test_ensemble_step_aggregation(ensemble: CMCEnsemble) -> None:
    """N колонок → errors shape (N, input_dim), states shape (N, state_dim)."""
    u = np.array([1.0, 2.0, 3.0])

    out = ensemble.step(u)

    assert out.errors.shape == (3, 3)
    assert out.states.shape == (3, 3)


def test_ensemble_step_shape_mismatch(ensemble: CMCEnsemble) -> None:
    """u.shape != (input_dim,) → ValueError."""
    u = np.array([1.0, 2.0])  # (2,) vs (3,)

    with pytest.raises(ValueError):
        ensemble.step(u)


def test_ensemble_active(ensemble: CMCEnsemble) -> None:
    """Колонки с ненулевой ошибкой → active считает правильно.

    Note: EMA-сходимость приближается к 0, но не достигает точно 0 в float64
    (остаток ~1e-10). Поэтому для проверки "сходилась → не активна" нужен
    ненулевой порог. Дефолт 0.0 = "любая ненулевая ошибка активна" — корректен
    для первого шага, но не для сходимости.
    """
    u = np.array([1.0, 2.0, 3.0])

    out = ensemble.step(u)

    # Первый шаг из нулей: все 3 колонки получили ошибку u - α·u ≠ 0
    assert out.active == 3

    # Стабильный вход 200 шагов с порогом 1e-8 → ошибки < порога → active → 0
    converged = CMCEnsemble(
        columns=[
            ColumnConfig(input_dim=3, state_dim=3, specialization="tone"),
            ColumnConfig(input_dim=3, state_dim=3, specialization="rhythm"),
            ColumnConfig(input_dim=3, state_dim=3, specialization="meaning"),
        ],
        active_threshold=1e-8,
    )
    for _ in range(200):
        out = converged.step(u)

    assert out.active == 0


def test_ensemble_reset(ensemble: CMCEnsemble) -> None:
    """После reset все состояния нулевые, active == 0."""
    u = np.array([1.0, 2.0, 3.0])
    ensemble.step(u)

    ensemble.reset()

    assert ensemble.active == 0
    # Следующий step из нулей снова даёт активные колонки
    out = ensemble.step(u)
    assert out.active == 3


def test_ensemble_active_before_step() -> None:
    """active property == 0 до первого step()."""
    ensemble = CMCEnsemble(columns=[ColumnConfig(input_dim=3, state_dim=3)])
    assert ensemble.active == 0


def test_ensemble_default_threshold() -> None:
    """active_threshold по умолчанию = 1e-8 (EMA never converges to exact 0.0)."""
    ensemble = CMCEnsemble(columns=[ColumnConfig(input_dim=3, state_dim=3)])
    assert ensemble.active_threshold == pytest.approx(1e-8)


def test_ensemble_active_threshold() -> None:
    """active_threshold > 0: только колонки с ||e||² > threshold активны."""
    ensemble = CMCEnsemble(
        columns=[ColumnConfig(input_dim=1, state_dim=1, alpha=0.5)],
        active_threshold=0.5,
    )
    u = np.array([1.0])
    out = ensemble.step(u)
    # e = u - x = 1.0 - 0.5*1.0 = 0.5; ||e||² = 0.25 < 0.5 → не активна
    assert out.active == 0

    ensemble.reset()
    u = np.array([2.0])
    out = ensemble.step(u)
    # e = 2.0 - 1.0 = 1.0; ||e||² = 1.0 > 0.5 → активна
    assert out.active == 1


# --- Проекции reads (ADR-0013) ---------------------------------------------

SEGMENTS = {"a": 2, "b": 1}


def test_ensemble_reads_require_segment_dims() -> None:
    """reads у колонки без segment_dims → ValueError (нельзя разрешить)."""
    with pytest.raises(ValueError):
        CMCEnsemble(
            columns=[
                ColumnConfig(input_dim=3, state_dim=3, reads=("a",)),
            ]
        )


def test_ensemble_reads_unknown_segment_fail_fast() -> None:
    """Неизвестный сегмент в reads → ValueError при сборке."""
    with pytest.raises(ValueError):
        CMCEnsemble(
            columns=[ColumnConfig(input_dim=3, state_dim=3, reads=("banana",))],
            segment_dims=SEGMENTS,
        )


def test_ensemble_reads_empty_fail_fast() -> None:
    """Пустой reads → ValueError (ловится в ColumnConfig)."""
    with pytest.raises(ValueError):
        ColumnConfig(input_dim=3, state_dim=3, reads=())


def test_ensemble_reads_selective_state() -> None:
    """Избирательность: возмущение сегмента меняет только читающие колонки."""
    ensemble = CMCEnsemble(
        columns=[
            ColumnConfig(input_dim=3, state_dim=3, specialization="a_only", reads=("a",)),
            ColumnConfig(input_dim=3, state_dim=3, specialization="b_only", reads=("b",)),
        ],
        segment_dims=SEGMENTS,
    )
    # Возмущаем только сегмент a ([0:2]); b=[2] остаётся 0.
    u = np.array([1.0, 2.0, 0.0])

    out = ensemble.step(u)

    # a_only читает a → ненулевая ошибка на [0:2]; b_only — нет.
    assert np.any(out.errors[0] != 0.0)
    assert np.all(out.errors[1] == 0.0)
    # b_only не «наблюдал» ноль → его состояние не обновлялось.
    np.testing.assert_array_equal(out.states[1], np.zeros(3))


def test_ensemble_reads_no_phantom_error_on_other_segment() -> None:
    """Возмущение нечитаемого сегмента не создаёт ошибку у колонки."""
    ensemble = CMCEnsemble(
        columns=[
            ColumnConfig(input_dim=3, state_dim=3, specialization="a_only", reads=("a",)),
        ],
        segment_dims=SEGMENTS,
    )
    # Возмущаем только b=[2]; a=[0:2] нулевые.
    u = np.array([0.0, 0.0, 5.0])

    out = ensemble.step(u)

    # a_only не читает b → ошибка ровно 0 (нет фиктивного «наблюдения нуля»).
    np.testing.assert_array_equal(out.errors[0], np.zeros(3))


def test_ensemble_reads_legacy_invariant() -> None:
    """reads=None → x, e, active совпадают с прежним поведением (legacy)."""
    columns = [
        ColumnConfig(input_dim=3, state_dim=3, specialization="tone"),
        ColumnConfig(input_dim=3, state_dim=3, specialization="rhythm"),
    ]
    legacy = CMCEnsemble(columns=columns)
    with_segments = CMCEnsemble(columns=columns, segment_dims=SEGMENTS)
    u = np.array([1.0, 2.0, 3.0])

    for _ in range(10):
        out_legacy = legacy.step(u)
        out_seg = with_segments.step(u)

    np.testing.assert_array_equal(out_legacy.errors, out_seg.errors)
    np.testing.assert_array_equal(out_legacy.states, out_seg.states)
    assert out_legacy.active == out_seg.active


def test_ensemble_resolved_reads_observability() -> None:
    """resolved_reads показывает фактические сегменты колонок (ADR-0013 §8)."""
    ensemble = CMCEnsemble(
        columns=[
            ColumnConfig(input_dim=3, state_dim=3, specialization="a_only", reads=("a",)),
            ColumnConfig(input_dim=3, state_dim=3, specialization="all"),
        ],
        segment_dims=SEGMENTS,
    )

    assert ensemble.resolved_reads == (("a",), ("a", "b"))


def test_ensemble_resolved_reads_without_segments() -> None:
    """Без segment_dims проекции не задействованы → пустые reads."""
    ensemble = CMCEnsemble(columns=[ColumnConfig(input_dim=3, state_dim=3)])

    assert ensemble.resolved_reads == ((),)


def test_ensemble_overlap_adds_contribution() -> None:
    """Перекрытие: дублирование колонки добавляет её вклад в ошибки (FFF)."""
    base = CMCEnsemble(
        columns=[
            ColumnConfig(input_dim=3, state_dim=3, specialization="a_only", reads=("a",)),
            ColumnConfig(input_dim=3, state_dim=3, specialization="b_only", reads=("b",)),
        ],
        segment_dims=SEGMENTS,
    )
    dup = CMCEnsemble(
        columns=[
            ColumnConfig(input_dim=3, state_dim=3, specialization="a_only", reads=("a",)),
            ColumnConfig(input_dim=3, state_dim=3, specialization="b_only", reads=("b",)),
            ColumnConfig(input_dim=3, state_dim=3, specialization="a_copy", reads=("a",)),
        ],
        segment_dims=SEGMENTS,
    )
    u = np.array([1.0, 2.0, 3.0])

    base_out = base.step(u)
    dup_out = dup.step(u)

    base_fff = float(np.sum(base_out.errors**2))
    dup_fff = float(np.sum(dup_out.errors**2))
    # Вклад a-колонки удвоился → FFF вырос ровно на её вклад.
    assert dup_fff == pytest.approx(base_fff + float(np.sum(base_out.errors[0] ** 2)))


def test_ensemble_winner_identity_by_specialization() -> None:
    """Разные возмущения → разные уникальные победители (по тегу, не индексу)."""
    columns = [
        ColumnConfig(input_dim=3, state_dim=3, specialization="a_only", reads=("a",)),
        ColumnConfig(input_dim=3, state_dim=3, specialization="b_only", reads=("b",)),
    ]

    def winner(ensemble: CMCEnsemble, u: np.ndarray) -> str:
        out = ensemble.step(u)
        activities = np.sum(out.errors**2, axis=1)
        return columns[int(np.argmax(activities))].specialization

    # Возмущаем b сильнее → побеждает b_only.
    ens_b = CMCEnsemble(columns=columns, segment_dims=SEGMENTS)
    assert winner(ens_b, np.array([0.1, 0.1, 5.0])) == "b_only"

    # Возмущаем a сильнее → побеждает a_only.
    ens_a = CMCEnsemble(columns=columns, segment_dims=SEGMENTS)
    assert winner(ens_a, np.array([5.0, 5.0, 0.1])) == "a_only"


def test_ensemble_winner_tag_stable_under_column_reorder() -> None:
    """Перестановка колонок сохраняет тег уникального победителя."""
    def winner_tag(order: list[ColumnConfig], u: np.ndarray) -> str:
        ensemble = CMCEnsemble(columns=order, segment_dims=SEGMENTS)
        out = ensemble.step(u)
        activities = np.sum(out.errors**2, axis=1)
        return order[int(np.argmax(activities))].specialization

    a = ColumnConfig(input_dim=3, state_dim=3, specialization="a_only", reads=("a",))
    b = ColumnConfig(input_dim=3, state_dim=3, specialization="b_only", reads=("b",))
    u = np.array([5.0, 5.0, 0.1])  # a сильнее → a_only

    assert winner_tag([a, b], u) == "a_only"
    assert winner_tag([b, a], u) == "a_only"


def test_ensemble_tie_breaks_by_index() -> None:
    """Ничья: при равных activities побеждает меньший индекс (правило)."""
    columns = [
        ColumnConfig(input_dim=3, state_dim=3, specialization="first", reads=("a",)),
        ColumnConfig(input_dim=3, state_dim=3, specialization="second", reads=("a",)),
    ]
    ensemble = CMCEnsemble(columns=columns, segment_dims=SEGMENTS)
    u = np.array([1.0, 1.0, 0.0])  # обе читают a одинаково → ничья

    out = ensemble.step(u)
    activities = np.sum(out.errors**2, axis=1)

    assert activities[0] == pytest.approx(activities[1])
    assert int(np.argmax(activities)) == 0
