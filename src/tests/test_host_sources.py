"""Unit tests for signal providers and the sensory bus (src/host/sources.py).

Covers determinism (same inputs → same SignalSource), categories,
severity → reflex contract, segment map, and the slow-tick cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.host.sources import (
    BatteryProvider,
    BusSegment,
    CircadianProvider,
    ConstantProvider,
    CpuProvider,
    NoisyProvider,
    SignalBus,
    StepProvider,
    UserMessageProvider,
    default_providers,
)
from src.mcp import SignalCategory


class TestCircadianProvider:
    """Циркадный провайдер: гармоники фазы суток."""

    def test_dim_and_category(self) -> None:
        provider = CircadianProvider()
        assert provider.dim == 2
        assert provider.category == SignalCategory.EXTEROCEPTIVE

    def test_midnight_is_unit_vector(self) -> None:
        """now=0 → полночь → [sin 0, cos 0] = [0, 1]."""
        signal = CircadianProvider().read(tick=0, now=0.0)
        np.testing.assert_allclose(signal.data, [0.0, 1.0], atol=1e-12)

    def test_noon_is_opposite(self) -> None:
        """12 часов → [sin π, cos π] = [0, -1]."""
        signal = CircadianProvider().read(tick=0, now=12 * 3600.0)
        np.testing.assert_allclose(signal.data, [0.0, -1.0], atol=1e-12)

    def test_deterministic(self) -> None:
        a = CircadianProvider().read(3, 100.0)
        b = CircadianProvider().read(3, 100.0)
        np.testing.assert_array_equal(a.data, b.data)


class TestBatteryProvider:
    """Батарея: разряд и severity → reflex."""

    def test_dim_and_category(self) -> None:
        provider = BatteryProvider()
        assert provider.dim == 1
        assert provider.category == SignalCategory.INTEROCEPTIVE

    def test_full_at_start(self) -> None:
        signal = BatteryProvider().read(tick=0, now=0.0)
        assert signal.data[0] == 1.0
        assert signal.severity == 0.0
        assert signal.is_reflex is False

    def test_drains_linearly(self) -> None:
        signal = BatteryProvider(start_level=1.0, drain_per_tick=0.01).read(
            tick=50, now=0.0
        )
        assert signal.data[0] == pytest.approx(0.5)
        assert signal.severity == pytest.approx(0.5)

    def test_never_below_zero(self) -> None:
        signal = BatteryProvider(drain_per_tick=0.1).read(tick=1000, now=0.0)
        assert signal.data[0] == 0.0
        assert signal.severity == 1.0

    def test_reflex_at_critical_level(self) -> None:
        """level ≤ 0.1 → severity ≥ 0.9 → is_reflex=True."""
        signal = BatteryProvider(start_level=1.0, drain_per_tick=0.01).read(
            tick=90, now=0.0
        )
        assert signal.severity >= 0.9
        assert signal.is_reflex is True

    def test_deterministic(self) -> None:
        a = BatteryProvider().read(10, 0.0)
        b = BatteryProvider().read(10, 0.0)
        assert a.data[0] == b.data[0]
        assert a.severity == b.severity


class TestCpuProvider:
    """CPU: детерминированный шум, severity = load."""

    def test_dim_and_category(self) -> None:
        provider = CpuProvider()
        assert provider.dim == 1
        assert provider.category == SignalCategory.INTEROCEPTIVE

    def test_load_within_bounds(self) -> None:
        provider = CpuProvider()
        for tick in range(50):
            load = provider.read(tick, 0.0).data[0]
            assert 0.0 <= load <= 1.0

    def test_severity_equals_load(self) -> None:
        signal = CpuProvider().read(tick=7, now=0.0)
        assert signal.severity == pytest.approx(signal.data[0])

    def test_deterministic_same_seed(self) -> None:
        a = CpuProvider(seed=42).read(5, 0.0)
        b = CpuProvider(seed=42).read(5, 0.0)
        np.testing.assert_array_equal(a.data, b.data)

    def test_different_seed_differs(self) -> None:
        a = CpuProvider(seed=1).read(5, 0.0)
        b = CpuProvider(seed=2).read(5, 0.0)
        assert a.data[0] != b.data[0]


class TestUserMessageProvider:
    """Коммуникативный сигнал: фиксированный вектор-заглушка."""

    def test_dim_and_category(self) -> None:
        provider = UserMessageProvider(embedding_dim=16)
        assert provider.dim == 16
        assert provider.category == SignalCategory.COMMUNICATIVE

    def test_shape(self) -> None:
        signal = UserMessageProvider(embedding_dim=8).read(tick=0, now=0.0)
        assert signal.data.shape == (8,)

    def test_stable_across_ticks(self) -> None:
        """Сообщение не меняется между тиками — вектор зафиксирован зерном."""
        provider = UserMessageProvider(seed=7)
        np.testing.assert_array_equal(
            provider.read(0, 0.0).data, provider.read(99, 500.0).data
        )


class TestConstantProvider:
    """Константа: тестовый генератор сходимости."""

    def test_returns_value(self) -> None:
        provider = ConstantProvider(value=(1.0, 2.0))
        signal = provider.read(tick=0, now=0.0)
        np.testing.assert_array_equal(signal.data, [1.0, 2.0])
        assert provider.dim == 2

    def test_unchanged_across_ticks(self) -> None:
        provider = ConstantProvider(value=(3.0,))
        np.testing.assert_array_equal(
            provider.read(0, 0.0).data, provider.read(50, 1e9).data
        )


class TestStepProvider:
    """Ступенька: before → after на заданном тике."""

    def test_before_step(self) -> None:
        provider = StepProvider(before=(0.0,), after=(5.0,), step_at=10)
        np.testing.assert_array_equal(provider.read(9, 0.0).data, [0.0])

    def test_after_step(self) -> None:
        provider = StepProvider(before=(0.0,), after=(5.0,), step_at=10)
        np.testing.assert_array_equal(provider.read(10, 0.0).data, [5.0])
        np.testing.assert_array_equal(provider.read(11, 0.0).data, [5.0])

    def test_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            StepProvider(before=(0.0,), after=(1.0, 2.0), step_at=1)

    def test_negative_step_at_raises(self) -> None:
        with pytest.raises(ValueError):
            StepProvider(before=(0.0,), after=(1.0,), step_at=-1)


class TestNoisyProvider:
    """Шум: детерминированный белый шум."""

    def test_shape(self) -> None:
        signal = NoisyProvider(dim=4).read(tick=0, now=0.0)
        assert signal.data.shape == (4,)

    def test_deterministic(self) -> None:
        a = NoisyProvider(dim=3, seed=5).read(2, 0.0)
        b = NoisyProvider(dim=3, seed=5).read(2, 0.0)
        np.testing.assert_array_equal(a.data, b.data)

    def test_ticks_differ(self) -> None:
        provider = NoisyProvider(dim=3, seed=5)
        assert not np.array_equal(
            provider.read(0, 0.0).data, provider.read(1, 0.0).data
        )

    def test_invalid_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            NoisyProvider(dim=0)


class TestSignalBus:
    """Шина: карта сегментов, агрегация, медленный такт."""

    def test_bus_dim_is_sum_of_dims(self) -> None:
        bus = SignalBus(default_providers())
        assert bus.bus_dim == 12

    def test_segment_offsets_are_contiguous(self) -> None:
        bus = SignalBus(default_providers())
        offset = 0
        for segment in bus.segments:
            assert segment.offset == offset
            offset += segment.dim
        assert offset == bus.bus_dim

    def test_segment_names_match_providers(self) -> None:
        providers = default_providers()
        bus = SignalBus(providers)
        assert [s.name for s in bus.segments] == [p.tag for p in providers]

    def test_step_returns_bus_dim_vector(self) -> None:
        bus = SignalBus(default_providers())
        assert bus.step(0, 0.0).shape == (12,)

    def test_aggregate_matches_concat(self) -> None:
        """Шина == конкатенация сигналов последнего step()."""
        bus = SignalBus(default_providers())
        u = bus.step(0, 0.0)
        expected = np.concatenate([s.data for s in bus.last_signals])
        np.testing.assert_array_equal(u, expected)

    def test_deterministic(self) -> None:
        a = SignalBus(default_providers()).step(4, 123.0)
        b = SignalBus(default_providers()).step(4, 123.0)
        np.testing.assert_array_equal(a, b)

    def test_empty_providers_raises(self) -> None:
        with pytest.raises(ValueError):
            SignalBus([])

    def test_duplicate_tag_raises(self) -> None:
        with pytest.raises(ValueError):
            SignalBus(
                [
                    ConstantProvider(value=(1.0,), tag="dup"),
                    ConstantProvider(value=(2.0,), tag="dup"),
                ]
            )

    def test_slow_tick_reuses_cache(self) -> None:
        """period=3 → сигнал читается на тиках 0, 3, 6... (медленный такт)."""
        provider = NoisyProvider(dim=2, seed=1, period=3)
        bus = SignalBus([provider])
        u0 = bus.step(0, 0.0)
        u1 = bus.step(1, 0.0)  # cached
        u2 = bus.step(2, 0.0)  # cached
        u3 = bus.step(3, 0.0)  # fresh
        np.testing.assert_array_equal(u0, u1)
        np.testing.assert_array_equal(u1, u2)
        assert not np.array_equal(u2, u3)

    def test_segments_are_bus_segment(self) -> None:
        bus = SignalBus(default_providers())
        assert all(isinstance(s, BusSegment) for s in bus.segments)
