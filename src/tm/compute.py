"""Functional Core — pure partner matching and signature updates (S5).

Deterministic, side-effect-free. Identity is inferred by matching a new
utterance against accumulated signatures (ADR-0008): no ``speaker_id`` is ever
read or written. ``cosine_similarity`` is reused from ``memory`` — signatures
live in the shared-memory domain, not a separate store.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from src.memory.serialize import Vector
from src.memory.similarity import cosine_similarity

from .models import PartnerSignature, PartnerState


def normalize_pause(pause_s: float, tau_s: float) -> float:
    """Нормировать паузу диалога в [0, 1) (чистая, S5 §3).

    ``1 - exp(-pause_s / tau_s)``. Монотонна, насыщается к 1. Отрицательная
    пауза трактуется как 0.

    Args:
        pause_s: Интервал с прошлой реплики, с (>= 0).
        tau_s: Постоянная нормировки, с (> 0).

    Returns:
        Нормированная пауза ∈ [0, 1).

    Raises:
        ValueError: Если tau_s <= 0.
    """
    if tau_s <= 0.0:
        raise ValueError(f"tau_s must be > 0, got {tau_s}")
    pause_s = max(pause_s, 0.0)
    return float(1.0 - math.exp(-pause_s / tau_s))


def detect_conflict(
    claim_embedding: Vector,
    memory_embeddings: Sequence[Vector],
) -> float:
    """Рассогласование нового утверждения с накопленным (чистая, S5 §4).

    Конфликт = ``max(1 - cos(claim, memory))`` по накопленному: чем меньше
    утверждение похоже на всё, что уже известно, тем выше конфликт. Триггер
    Vigilance — конфликт, а не любое утверждение (решение D). Пустое
    накопленное → 0.0 (не с чем конфликтовать).

    Args:
        claim_embedding: Эмбеддинг нового утверждения.
        memory_embeddings: Эмбеддинги накопленного знания.

    Returns:
        Рассогласование ∈ [0, 1].

    Raises:
        ValueError: Если размерности не совпадают.
    """
    if not memory_embeddings:
        return 0.0
    worst = 0.0
    for memory in memory_embeddings:
        if memory.shape != claim_embedding.shape:
            raise ValueError(
                f"memory shape {memory.shape} != claim shape {claim_embedding.shape}"
            )
        similarity = cosine_similarity(claim_embedding, memory)
        worst = max(worst, 1.0 - similarity)
    return float(min(1.0, max(0.0, worst)))


def match_partner(
    embedding: Vector,
    signatures: Sequence[PartnerSignature],
    *,
    threshold: float,
) -> tuple[int | None, float]:
    """Сопоставить реплику с накопленными сигнатурами.

    Узнавание = близость выше порога. Тай-брейк — первый максимум (стабильный
    порядок). Пустой список → ``(None, 0.0)``.

    Args:
        embedding: Эмбеддинг новой реплики.
        signatures: Накопленные сигнатуры.
        threshold: Порог косинусной близости для «узнавания».

    Returns:
        ``(index, similarity)``: индекс лучшей сигнатуры или ``None``
        (не узнан), и её близость.

    Raises:
        ValueError: Если threshold вне [0, 1] или сигнатура не той размерности.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if not signatures:
        return None, 0.0

    best_index: int | None = None
    best_sim = -1.0
    for index, signature in enumerate(signatures):
        if signature.centroid.shape != embedding.shape:
            raise ValueError(
                f"signature centroid shape {signature.centroid.shape} "
                f"!= embedding shape {embedding.shape}"
            )
        similarity = cosine_similarity(embedding, signature.centroid)
        if similarity > best_sim:
            best_sim = similarity
            best_index = index

    if best_index is None or best_sim < threshold:
        return None, max(best_sim, 0.0)
    return best_index, best_sim


def update_signature(
    signature: PartnerSignature | None,
    embedding: Vector,
    *,
    pause_s: float,
    valence: float,
    stress: float,
    learning_rate: float,
) -> PartnerSignature:
    """Обновить (или создать) сигнатуру по новой реплике (чистая).

    Скользящее среднее ``centroid`` и ``mean_*``; ``weight`` растёт на 1.
    ``None`` → сигнатура создаётся с нулевым весом-стартом. Вход не мутируется.

    Args:
        signature: Текущая сигнатура или None (первая реплика партнёра).
        embedding: Эмбеддинг новой реплики.
        pause_s: Интервал с прошлой реплики, с.
        valence: Реакция хоста (валентность).
        stress: Стресс в момент реплики.
        learning_rate: Скорость обновления центроида, (0, 1].

    Returns:
        Новая сигнатура (frozen).

    Raises:
        ValueError: Если learning_rate вне (0, 1] или pause_s < 0.
    """
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
    if pause_s < 0.0:
        raise ValueError(f"pause_s must be >= 0, got {pause_s}")

    new_embedding = np.asarray(embedding, dtype=np.float64)
    if signature is None:
        return PartnerSignature(
            centroid=new_embedding.copy(),
            weight=1.0,
            mean_pause_s=float(pause_s),
            mean_valence=float(valence),
            mean_stress=float(stress),
        )

    old = np.asarray(signature.centroid, dtype=np.float64)
    if old.shape != new_embedding.shape:
        raise ValueError(
            f"centroid shape {old.shape} != embedding shape {new_embedding.shape}"
        )
    centroid = old + learning_rate * (new_embedding - old)
    return PartnerSignature(
        centroid=centroid,
        weight=signature.weight + 1.0,
        mean_pause_s=signature.mean_pause_s
        + learning_rate * (pause_s - signature.mean_pause_s),
        mean_valence=signature.mean_valence
        + learning_rate * (valence - signature.mean_valence),
        mean_stress=signature.mean_stress
        + learning_rate * (stress - signature.mean_stress),
        name=signature.name,
        aliases=signature.aliases,
    )


def update_trust(
    state: PartnerState,
    *,
    conflict: float,
    ambiguity: float,
    trust_gain: float,
    trust_decay: float,
) -> PartnerState:
    """Обновить доверие/двусмысленность/конфликт по итогу реплики (чистая).

    Доверие растёт при согласии (низкий конфликт) и утекает в его отсутствие:
    ``trust' = clamp(trust - trust_decay + trust_gain·(1 - conflict))``.
    Конфликт/двусмысленность фиксируются как есть (clamp в [0, 1]).

    Args:
        state: Предыдущее состояние партнёра.
        conflict: Рассогласование текущей реплики, [0, 1].
        ambiguity: Двусмысленность текущей реплики, [0, 1].
        trust_gain: Прирост доверия при согласии, ≥ 0.
        trust_decay: Утечка доверия, ≥ 0.

    Returns:
        Новое состояние (frozen).

    Raises:
        ValueError: Если conflict/ambiguity вне [0, 1] или gain/decay < 0.
    """
    if not 0.0 <= conflict <= 1.0:
        raise ValueError(f"conflict must be in [0, 1], got {conflict}")
    if not 0.0 <= ambiguity <= 1.0:
        raise ValueError(f"ambiguity must be in [0, 1], got {ambiguity}")
    if trust_gain < 0.0 or trust_decay < 0.0:
        raise ValueError(
            f"trust_gain/trust_decay must be >= 0, got {trust_gain}/{trust_decay}"
        )

    trust = state.trust - trust_decay + trust_gain * (1.0 - conflict)
    trust = min(1.0, max(0.0, trust))
    return PartnerState(
        trust=trust,
        ambiguity=ambiguity,
        conflict=conflict,
        uncertainty=state.uncertainty,
        name=state.name,
    )
