"""Unit tests for ResourceMeter and ResourceProvider (host resources)."""

from __future__ import annotations

import numpy as np
import pytest

from src.host.resources import ResourceMeter, ResourceProvider
from src.mcp import SignalCategory


class FakeMeter(ResourceMeter):
    """Фейковый meter: детерминированные метрики для тестов."""

    def __init__(self, latency_s: float = 0.0, rss_mb: float = 0.0) -> None:
        super().__init__()
        self._fake_latency = latency_s
        self._fake_rss = rss_mb

    @property
    def last_latency_s(self) -> float:
        return self._fake_latency

    @property
    def last_rss_mb(self) -> float:
        return self._fake_rss


class TestResourceMeter:
    """Meter: запись тика и чтение RSS."""

    def test_initial_zero(self) -> None:
        meter = ResourceMeter()
        assert meter.last_latency_s == 0.0
        assert meter.last_rss_mb == 0.0

    def test_record_tick(self) -> None:
        meter = ResourceMeter()
        meter.record_tick(0.005)
        assert meter.last_latency_s == pytest.approx(0.005)
        assert meter.last_rss_mb > 0.0  # реальный RSS процесса

    def test_negative_latency_clamped(self) -> None:
        meter = ResourceMeter()
        meter.record_tick(-1.0)
        assert meter.last_latency_s == 0.0

    def test_current_rss_positive(self) -> None:
        assert ResourceMeter.current_rss_mb() > 0.0


class TestResourceProvider:
    """Provider: нормировка, severity, reflex."""

    def test_dim_and_category(self) -> None:
        provider = ResourceProvider(meter=FakeMeter())
        assert provider.dim == 2
        assert provider.category == SignalCategory.INTEROCEPTIVE

    def test_normal_load_low_severity(self) -> None:
        """Нормальная нагрузка → низкий severity."""
        meter = FakeMeter(latency_s=0.001, rss_mb=100.0)
        provider = ResourceProvider(
            meter=meter, tick_budget_ms=50.0, rss_budget_mb=1024.0
        )
        signal = provider.read(tick=0, now=0.0)
        assert signal.data.shape == (2,)
        assert signal.severity < 0.5
        assert signal.is_reflex is False

    def test_latency_overload_high_severity(self) -> None:
        """Латентность выше бюджета → severity ≥ 1 (клип)."""
        meter = FakeMeter(latency_s=0.1, rss_mb=100.0)  # 100 мс при бюджете 50
        provider = ResourceProvider(meter=meter, tick_budget_ms=50.0)
        signal = provider.read(tick=0, now=0.0)
        assert signal.severity == pytest.approx(1.0)

    def test_reflex_at_critical_severity(self) -> None:
        """severity ≥ 0.9 → is_reflex=True (контракт SignalSource)."""
        meter = FakeMeter(latency_s=0.05, rss_mb=100.0)  # 50/50 = 1.0
        provider = ResourceProvider(meter=meter, tick_budget_ms=50.0)
        signal = provider.read(tick=0, now=0.0)
        assert signal.severity >= 0.9
        assert signal.is_reflex is True

    def test_rss_overload(self) -> None:
        """RSS выше бюджета → высокий severity."""
        meter = FakeMeter(latency_s=0.0, rss_mb=2000.0)
        provider = ResourceProvider(meter=meter, rss_budget_mb=1024.0)
        signal = provider.read(tick=0, now=0.0)
        assert signal.severity == pytest.approx(1.0)

    def test_data_norms(self) -> None:
        """data = [latency_norm, rss_norm]."""
        meter = FakeMeter(latency_s=0.025, rss_mb=512.0)  # 25/50, 512/1024
        provider = ResourceProvider(
            meter=meter, tick_budget_ms=50.0, rss_budget_mb=1024.0
        )
        signal = provider.read(tick=0, now=0.0)
        np.testing.assert_allclose(signal.data, [0.5, 0.5], atol=1e-9)

    def test_invalid_budget_raises(self) -> None:
        with pytest.raises(ValueError):
            ResourceProvider(meter=FakeMeter(), tick_budget_ms=0.0)
        with pytest.raises(ValueError):
            ResourceProvider(meter=FakeMeter(), rss_budget_mb=-1.0)
