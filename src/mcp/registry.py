"""Signal registry — Imperative Shell for managing signal sources.

Functional Core / Imperative Shell (ADR-0004):
- Models (models.py) — pure data, tested without Shell
- Registry (registry.py) — imperative shell, owns sources list
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .models import SignalCategory, SignalSource, Vector

logger = logging.getLogger(__name__)


class SignalRegistry:
    """Управление источниками сигналов и агрегация в единый вектор u(t).

    Imperative Shell: владеет списком зарегистрированных источников,
    агрегирует их в единый вектор для CMCEnsemble.step().

    Attributes:
        active_threshold: Порог активности колонки (передаётся в CMCEnsemble).
    """

    def __init__(self, active_threshold: float = 1e-8) -> None:
        """Создание registry с порогом активности.

        Args:
            active_threshold: Порог для CMCEnsemble.
                Дефолт 1e-8 — EMA never converges to exact 0.0 in float64.
        """
        self._active_threshold = active_threshold
        self._sources: list[SignalSource] = []

    def register(self, signal: SignalSource) -> None:
        """Добавить источник сигнала в registry.

        Args:
            signal: SignalSource для регистрации.
        """
        self._sources.append(signal)
        logger.debug(
            "Registered signal: tag=%s, category=%s, severity=%.2f",
            signal.tag,
            signal.category.value,
            signal.severity,
        )

    def register_many(self, signals: list[SignalSource]) -> None:
        """Пакетная регистрация нескольких источников.

        Args:
            signals: Список SignalSource для регистрации.
        """
        for signal in signals:
            self.register(signal)

    def unregister_by_tag(self, tag: str) -> bool:
        """Удалить источники по tag.

        Args:
            tag: Тег для удаления.

        Returns:
            True, если хотя бы один источник удалён.
        """
        before = len(self._sources)
        self._sources = [
            s for s in self._sources if s.tag != tag
        ]
        removed = before - len(self._sources)
        if removed > 0:
            logger.debug("Unregistered %d signals with tag=%s", removed, tag)
        return removed > 0

    def clear(self) -> None:
        """Удалить все источники."""
        self._sources.clear()
        logger.debug("Cleared all signal sources")

    @property
    def sources(self) -> list[SignalSource]:
        """Список зарегистрированных источников (read-only view)."""
        return list(self._sources)

    @property
    def count(self) -> int:
        """Количество зарегистрированных источников."""
        return len(self._sources)

    def get_by_category(self, category: SignalCategory) -> list[SignalSource]:
        """Фильтрация по категории.

        Args:
            category: Категория для фильтрации.

        Returns:
            Список источников указанной категории.
        """
        return [s for s in self._sources if s.category == category]

    def get_reflex_signals(self) -> list[SignalSource]:
        """Получить все reflex-сигналы.

        Returns:
            Список reflex-сигналов (только interoceptive с severity ≥ 0.9).
        """
        return [s for s in self._sources if s.is_reflex]

    def aggregate(self) -> Optional[Vector]:
        """Агрегировать все сигналы в единый вектор для CMCEnsemble.

        В Фазе 1: простая конкатенация всех векторов data.
        В Фазе 2+: проекции по специализациям.

        Returns:
            Агрегированный вектор shape=(total_dim,), или None если нет источников.
        """
        if not self._sources:
            return None

        data_vectors = [s.data for s in self._sources]
        return np.concatenate(data_vectors)

    @property
    def active_threshold(self) -> float:
        """Порог активности колонок."""
        return self._active_threshold
