"""Memory consolidation — active pruning + Structure Learning (S6).

Sleep-time consolidation (manifest §3.Д, §4): prune unimportant episodes and
generalise recurring patterns into *schemas*. The plan is a pure function; the
execution is a shell. Every deletion is explicit and logged — this preserves
invariant 6 (recall is monotone **except for explicit consolidation**,
ADR-0009 §4). Deterministic, no LLM (ADR-0007).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .models import Episode
from .protocols import SupportsConsolidate
from .serialize import Vector
from .similarity import cosine_similarity

logger = logging.getLogger(__name__)

_EPS = 1e-12


def episode_weight(
    episode: Episode,
    *,
    now: float,
    recency_tau_s: float,
) -> float:
    """Вес эпизода для pruning, >= 0 (чистая).

    Комбинирует аффективную значимость (|valence| + stress) и свежесть
    (``exp(-age/recency_tau)``): значимые и свежие эпизоды ценнее.

    Args:
        episode: Эпизод.
        now: Текущее время (для давности).
        recency_tau_s: Постоянная свежести, с (> 0).

    Returns:
        Вес >= 0.

    Raises:
        ValueError: Если ``recency_tau_s <= 0``.
    """
    if recency_tau_s <= 0.0:
        raise ValueError(f"recency_tau_s must be > 0, got {recency_tau_s}")
    age = max(0.0, now - episode.timestamp)
    recency = float(np.exp(-age / recency_tau_s))
    affect = abs(episode.valence) + episode.stress
    return float((1.0 + affect) * recency)


@dataclass(frozen=True)
class Schema:
    """Схема — обобщение кластера похожих эпизодов (Structure Learning).

    Attributes:
        centroid: Усреднённый эмбеддинг кластера.
        member_count: Число эпизодов в кластере.
        summary: Резюме (первый/самый свежий content кластера).
    """

    centroid: Vector
    member_count: int
    summary: str


@dataclass(frozen=True)
class ConsolidationTrigger:
    """Решение о запуске ночного цикла (чистое).

    Attributes:
        due: Пора ли консолидировать.
        reason: Причина (для логирования/трассировки).
    """

    due: bool
    reason: str


def should_consolidate(
    *,
    tick: int,
    last_tick: int,
    episode_count: int,
    every_ticks: int,
    min_episodes: int,
) -> ConsolidationTrigger:
    """Решить, пора ли запускать ночной цикл (чистая, S6 проход 2).

    Триггер — по расписанию (``every_ticks``) и объёму (``min_episodes``).
    Консолидация остаётся явной и логируемой (инвариант 6); функция лишь
    выбирает момент.

    Args:
        tick: Текущий тик.
        last_tick: Тик последней консолидации.
        episode_count: Число эпизодов в памяти.
        every_ticks: Интервал ночного цикла, тики (<= 0 → выключен).
        min_episodes: Минимум эпизодов для срабатывания.

    Returns:
        ConsolidationTrigger.

    Raises:
        ValueError: Если ``tick < 0``, ``min_episodes < 0`` или
            ``episode_count < 0``.
    """
    if tick < 0:
        raise ValueError(f"tick must be >= 0, got {tick}")
    if min_episodes < 0:
        raise ValueError(f"min_episodes must be >= 0, got {min_episodes}")
    if episode_count < 0:
        raise ValueError(f"episode_count must be >= 0, got {episode_count}")
    if every_ticks <= 0:
        return ConsolidationTrigger(False, "night cycle disabled")
    if tick - last_tick < every_ticks:
        return ConsolidationTrigger(False, "interval not reached")
    if episode_count < min_episodes:
        return ConsolidationTrigger(False, "not enough episodes")
    return ConsolidationTrigger(True, f"night cycle at tick {tick}")


@dataclass(frozen=True)
class ConsolidationPlan:
    """План консолидации (чистый, до исполнения).

    Attributes:
        prune_ids: id эпизодов к удалению (низкий вес).
        schemas: Схемы (обобщения) для записи.
        kept: Сколько эпизодов сохранено (не удалено).
        reason: Человекочитаемая причина плана.
    """

    prune_ids: tuple[int, ...]
    schemas: tuple[Schema, ...]
    kept: int
    reason: str


@dataclass(frozen=True)
class ConsolidationResult:
    """Итог исполнения консолидации (Shell).

    Attributes:
        pruned: Сколько эпизодов удалено.
        schemas_saved: Сколько схем записано.
        reason: Причина (из плана).
    """

    pruned: int
    schemas_saved: int
    reason: str


def plan_consolidation(
    episodes: Sequence[Episode],
    *,
    min_weight: float,
    schema_threshold: float,
    max_schemas: int,
    now: float,
    recency_tau_s: float = 86_400.0,
) -> ConsolidationPlan:
    """Построить план pruning + Structure Learning (чистая).

    Pruning: эпизоды с весом ниже ``min_weight`` → удаление (незначимые и
    старые). Structure Learning: жадная кластеризация **сохранённых**
    эпизодов по косинусу (порог ``schema_threshold``), до ``max_schemas``
    схем; схема — усреднённый центроид кластера.

    Args:
        episodes: Все эпизоды (id заполнен).
        min_weight: Порог веса: ниже → pruning, >= 0.
        schema_threshold: Порог косинуса для объединения в схему, [0, 1].
        max_schemas: Максимум схем, >= 0.
        now: Текущее время (давность).
        recency_tau_s: Постоянная свежести, с (> 0).

    Returns:
        ConsolidationPlan.

    Raises:
        ValueError: Если параметры вне границ.
    """
    if min_weight < 0.0:
        raise ValueError(f"min_weight must be >= 0, got {min_weight}")
    if not 0.0 <= schema_threshold <= 1.0:
        raise ValueError(
            f"schema_threshold must be in [0, 1], got {schema_threshold}"
        )
    if max_schemas < 0:
        raise ValueError(f"max_schemas must be >= 0, got {max_schemas}")

    prune_ids: list[int] = []
    kept: list[Episode] = []
    for episode in episodes:
        weight = episode_weight(episode, now=now, recency_tau_s=recency_tau_s)
        if weight < min_weight:
            if episode.id is not None:
                prune_ids.append(episode.id)
        else:
            kept.append(episode)

    schemas = _build_schemas(kept, schema_threshold, max_schemas)
    reason = (
        f"pruned {len(prune_ids)} of {len(episodes)} episodes "
        f"(weight < {min_weight}); built {len(schemas)} schemas"
    )
    return ConsolidationPlan(
        prune_ids=tuple(prune_ids),
        schemas=tuple(schemas),
        kept=len(kept),
        reason=reason,
    )


def _build_schemas(
    episodes: Sequence[Episode],
    threshold: float,
    max_schemas: int,
) -> list[Schema]:
    """Жадная кластеризация эпизодов по косинусу (чистая).

    Args:
        episodes: Сохранённые эпизоды.
        threshold: Порог косинуса для присоединения к кластеру.
        max_schemas: Максимум кластеров.

    Returns:
        Список Schema (центроид + число участников + резюме).
    """
    if max_schemas == 0 or not episodes:
        return []
    clusters: list[list[Episode]] = []
    for episode in episodes:
        placed = False
        for cluster in clusters:
            centroid = np.mean([e.embedding for e in cluster], axis=0)
            if cosine_similarity(episode.embedding, centroid) >= threshold:
                cluster.append(episode)
                placed = True
                break
        if not placed:
            if len(clusters) < max_schemas:
                clusters.append([episode])
            else:
                # Нет места под новую схему → присоединить к ближайшему кластеру.
                best_idx = max(
                    range(len(clusters)),
                    key=lambda i: cosine_similarity(
                        episode.embedding,
                        np.mean([e.embedding for e in clusters[i]], axis=0),
                    ),
                )
                clusters[best_idx].append(episode)

    schemas: list[Schema] = []
    for cluster in clusters:
        centroid = np.mean([e.embedding for e in cluster], axis=0)
        # Резюме — content самого свежего участника.
        freshest = max(cluster, key=lambda e: e.timestamp)
        schemas.append(
            Schema(
                centroid=np.asarray(centroid, dtype=np.float64),
                member_count=len(cluster),
                summary=freshest.content,
            )
        )
    return schemas


def consolidate(
    store: SupportsConsolidate,
    *,
    min_weight: float,
    schema_threshold: float,
    max_schemas: int,
    now: float,
    recency_tau_s: float = 86_400.0,
) -> ConsolidationResult:
    """Исполнить консолидацию: pruning + сохранение схем (Shell).

    Читает эпизоды, строит план, удаляет помеченные и сохраняет схемы.
    **Явная консолидация** (инвариант 6): число удалённых логируется.

    Args:
        store: Хранилище с ``all_episodes``/``delete``/``save_schema``.
        min_weight: Порог веса для pruning.
        schema_threshold: Порог косинуса для схем.
        max_schemas: Максимум схем.
        now: Текущее время.
        recency_tau_s: Постоянная свежести, с.

    Returns:
        ConsolidationResult.
    """
    episodes = store.all_episodes()
    plan = plan_consolidation(
        episodes,
        min_weight=min_weight,
        schema_threshold=schema_threshold,
        max_schemas=max_schemas,
        now=now,
        recency_tau_s=recency_tau_s,
    )
    pruned = store.delete(list(plan.prune_ids))
    schemas_saved = 0
    for schema in plan.schemas:
        store.save_schema(schema.centroid, schema.member_count, schema.summary)
        schemas_saved += 1
    logger.info(
        "consolidation: pruned %d episodes, saved %d schemas (%s)",
        pruned,
        schemas_saved,
        plan.reason,
    )
    return ConsolidationResult(
        pruned=pruned, schemas_saved=schemas_saved, reason=plan.reason
    )
