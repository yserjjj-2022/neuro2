"""ResourceProvider — интероцептивная обратная связь по ресурсам.

«Тахикардия/одышка» хоста: латентность тика и RSS процесса. При перегрузке
severity растёт; при severity ≥ 0.9 сигнал помечается is_reflex (S4 — реакция,
S1 — только измерение и логирование).

Это единственный недетерминированный провайдер (реальный мир). В тестах
``ResourceMeter`` фейкается инъекцией.
"""

from __future__ import annotations

import resource
from dataclasses import dataclass

import numpy as np

from src.mcp import SignalCategory, SignalSource


class ResourceMeter:
    """Собирает метрики ресурсов процесса (Imperative Shell).

    Attributes:
        _last_latency_s: Длительность последнего тика, с.
        _last_rss_mb: RSS процесса на последнем измерении, МБ.
    """

    def __init__(self) -> None:
        self._last_latency_s: float = 0.0
        self._last_rss_mb: float = 0.0

    def record_tick(self, latency_s: float) -> None:
        """Зафиксировать длительность тика и текущий RSS.

        Args:
            latency_s: Длительность тика в секундах (≥ 0).
        """
        self._last_latency_s = max(0.0, latency_s)
        self._last_rss_mb = self.current_rss_mb()

    @property
    def last_latency_s(self) -> float:
        """Длительность последнего тика, с."""
        return self._last_latency_s

    @property
    def last_rss_mb(self) -> float:
        """RSS на последнем record_tick, МБ."""
        return self._last_rss_mb

    @staticmethod
    def current_rss_mb() -> float:
        """Текущий пиковый RSS процесса, МБ (stdlib resource).

        Returns:
            RSS в мегабайтах. На Linux ru_maxrss в КБ, на macOS — в байтах.
        """
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS отдаёт байты, Linux — килобайты. Нормируем по платформе.
        import sys

        if sys.platform == "darwin":
            return usage / (1024.0 * 1024.0)
        return usage / 1024.0


@dataclass(frozen=True)
class ResourceProvider:
    """Интероцептивный сигнал ресурсов: латентность + RSS.

    Attributes:
        meter: Источник измерений (инъекция; в тестах — фейк).
        tick_budget_ms: Бюджет длительности тика, мс.
        rss_budget_mb: Бюджет памяти, МБ.
        tag: Идентификатор источника.
        category: Интероцептивный — состояние самой системы.
        dim: Размерность: [latency_norm, rss_norm] = 2.
        period: Обновляется каждый тик.
    """

    meter: ResourceMeter
    tick_budget_ms: float = 50.0
    rss_budget_mb: float = 1024.0
    tag: str = "resources"
    category: SignalCategory = SignalCategory.INTEROCEPTIVE
    dim: int = 2
    period: int = 1

    def __post_init__(self) -> None:
        if self.tick_budget_ms <= 0.0:
            raise ValueError(f"tick_budget_ms must be > 0, got {self.tick_budget_ms}")
        if self.rss_budget_mb <= 0.0:
            raise ValueError(f"rss_budget_mb must be > 0, got {self.rss_budget_mb}")

    def read(self, tick: int, now: float) -> SignalSource:
        """Прочитать ресурсный сигнал на данном тике.

        Args:
            tick: Номер тика (не используется — метрики post-hoc).
            now: Wall-clock время (не используется).

        Returns:
            SignalSource shape=(2,) с [latency_norm, rss_norm];
            severity = max(latency_norm, rss_norm), клип [0, 1].
        """
        latency_norm = (self.meter.last_latency_s * 1000.0) / self.tick_budget_ms
        rss_norm = self.meter.last_rss_mb / self.rss_budget_mb
        data = np.array([latency_norm, rss_norm], dtype=np.float64)
        severity = float(np.clip(max(latency_norm, rss_norm, 0.0), 0.0, 1.0))
        return SignalSource(
            category=self.category,
            data=data,
            severity=severity,
            tag=self.tag,
        )
