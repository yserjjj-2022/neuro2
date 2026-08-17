"""Domain objects for MCP signal taxonomy.

Frozen dataclass + enum — аналог FreeEnergyResult (energy), VotingResult (voting),
TaskAttraction (attractors). Type aliases follow the idiom from
src/core/cmc/models.py — numpy generics are invariant, so Vector
must be explicitly typed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

# Любой float-вектор: float32, float64 и т.д.
Vector = np.ndarray[Any, np.dtype[np.floating[Any]]]


class SignalCategory(Enum):
    """Категория входящего сигнала.

    Три принципиально разнородных категории из манифеста §3.З:
    - exteroceptive — контекст внешнего мира (погода, локация, MCP-ресурсы)
    - interoceptive — внутреннее состояние системы (батарея, CPU, гомеостатика)
    - communicative — текст собеседника, сообщения
    """

    EXTEROCEPTIVE = "exteroceptive"
    INTEROCEPTIVE = "interoceptive"
    COMMUNICATIVE = "communicative"


@dataclass(frozen=True)
class SignalSource:
    """Единственный контракт источника сигнала.

    Каждый сигнал несёт категорию, вектор данных и метаданные.
    Разные категории обрабатываются разными маршрутами в host loop.

    Attributes:
        category: Категория сигнала (extero-/intero-/communicative).
        data: Вектор данных (shape зависит от категории и колонки).
            - exteroceptive: проекция в общее пространство D
            - interoceptive: скалярные/векторные маркеры (батарея, CPU)
            - communicative: эмбеддинг текста
        severity: От 0.0 до 1.0, где 1.0 — критический сигнал.
            interoceptive с severity ≥ 0.9 проходит через reflex-path.
        is_reflex: Быстрый обходной путь, минуя EMA-сглаживание и dwell.
            Только для критических interoceptive сигналов.
        tag: Строка-тег для идентификации источника
            (напр. "battery", "weather", "user_message").
    """

    category: SignalCategory
    data: Vector
    severity: float = 0.0
    is_reflex: bool = False
    tag: str = ""

    def __post_init__(self) -> None:
        """Валидация инвариантов после создания."""
        if self.severity < 0.0 or self.severity > 1.0:
            raise ValueError(
                f"severity must be in [0.0, 1.0], got {self.severity}"
            )
        if self.category == SignalCategory.INTEROCEPTIVE and self.severity >= 0.9:
            # Критические interoceptive сигналы — reflex по умолчанию
            object.__setattr__(self, "is_reflex", True)
        if self.is_reflex and self.category != SignalCategory.INTEROCEPTIVE:
            raise ValueError(
                "is_reflex=True only allowed for interoceptive signals"
            )
