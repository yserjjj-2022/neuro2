"""Memory prior — pure core encoding a recalled episode into bus channels.

The prior is a small bounded vector appended to the sensory bus, so a recalled
episode influences F(t) (and therefore behaviour) through the same path as any
other input. See ``stages/S2_SPEC.md`` §4.
"""

from __future__ import annotations

import numpy as np

from .models import Episode
from .serialize import Vector

# [cos(query, episode), tanh(valence), tanh(stress), tanh(free_energy)]
MEMORY_PRIOR_DIM = 4


def encode_memory_prior(
    episode: Episode | None,
    query: Vector | None,
) -> Vector:
    """Кодировать воспоминание в ограниченный вектор-приор.

    Формат (``MEMORY_PRIOR_DIM`` = 4):
        [cos(query, episode.embedding), tanh(valence), tanh(stress), tanh(f)]

    ``tanh`` держит аффективные компоненты в [-1, 1] — защита от blow-up F.
    Нет эпизода / нет запроса / нулевая норма → вектор нулей (память «молчит»).

    Args:
        episode: Извлечённый эпизод (None → нули).
        query: Вектор запроса (None → нули).

    Returns:
        Вектор shape=(MEMORY_PRIOR_DIM,) float64, конечный, ∈ [-1, 1].
    """
    prior = np.zeros(MEMORY_PRIOR_DIM, dtype=np.float64)
    if episode is None or query is None:
        return prior

    q = np.asarray(query, dtype=np.float64)
    e = np.asarray(episode.embedding, dtype=np.float64)
    if q.shape != e.shape:
        raise ValueError(f"query shape {q.shape} != episode embedding {e.shape}")

    q_norm = float(np.linalg.norm(q))
    e_norm = float(np.linalg.norm(e))
    cos = 0.0
    if q_norm > 0.0 and e_norm > 0.0:
        cos = float(np.dot(q, e) / (q_norm * e_norm))

    prior[0] = cos
    prior[1] = np.tanh(episode.valence)
    prior[2] = np.tanh(episode.stress)
    prior[3] = np.tanh(episode.free_energy)
    return prior
