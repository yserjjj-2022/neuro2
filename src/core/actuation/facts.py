"""Реестр фактов (S8 этап 2, ADR-0012).

**Единый источник имён фактов.** Core владеет смыслом и дефолтом факта, Shell
измеряет значение и отдаёт снимок ``Mapping[str, float]``. Явный реестр
фиксирует множество имён в одном месте: иначе Core и Shell разойдутся молча, и
оценка всегда пойдёт по дефолту — тихий баг.

Словарь **открыт**: реестр перечисляет известные факты, но ``evaluate_fact``
работает с любым ``Fact``; новый факт вводится добавлением записи (без правки
типов). Значения ∈ [0, 1]; категориальные факты (режим) здесь не тащим.
"""

from __future__ import annotations

from collections.abc import Mapping

from .models import DEFAULT_FACT_VALUE, Fact

# Начальный набор. Расширяется при связывании фактов с опциями (этап 3).
NETWORK_AVAILABLE = Fact(
    name="network_available",
    default=DEFAULT_FACT_VALUE,
    description="Есть доступ к сети (нужен внешним тулам).",
)
TOPIC_BOUND = Fact(
    name="topic_bound",
    default=DEFAULT_FACT_VALUE,
    description="Сообщение привязано к теме тула (привязка темы).",
)

FACTS: Mapping[str, Fact] = {
    NETWORK_AVAILABLE.name: NETWORK_AVAILABLE,
    TOPIC_BOUND.name: TOPIC_BOUND,
}
