"""Precision weighting γ — Functional Core + Imperative Shell.

γ = обратная дисперсия входного сигнала (манифест §3.Б): доверие каналу.
Стабильный канал → высокая γ; шумный канал → низкая γ.

Functional Core / Imperative Shell (ADR-0004):
- ``inverse_variance`` — чистая функция, тестируется без Shell
- ``PrecisionEstimator`` — Shell, владеет окном наблюдений
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.core.cmc.models import Vector


def inverse_variance(
    samples: Vector,
    eps: float = 1e-6,
    gamma_max: float = 10.0,
) -> Vector:
    """Точность γᵢ = clip(1 / (varᵢ + eps), 0, gamma_max).

    Чистая функция: не мутирует samples.

    Args:
        samples: Наблюдения формы (n_samples, dim).
        eps: Регуляризация знаменателя (защита от деления на ноль).
        gamma_max: Верхняя граница γ — «во сколько раз максимум доверяем
            каналу». Дефолт 10.0 (стабильный канал весит ≤10× шумного);
            1e6 вызывал взрыв F/stress (см. stages/S1_SPEC.md).

    Returns:
        γ формы (dim,).

    Raises:
        ValueError: Если samples пустой или не двумерный.

    Note:
        Один сэмпл → var = 0 → γ = gamma_max (максимальное доверие).
        Дисперсия считается по оси наблюдений (axis=0), population (ddof=0).
    """
    if samples.ndim != 2:
        raise ValueError(f"samples must be 2D (n_samples, dim), got {samples.ndim}D")
    if samples.shape[0] == 0:
        raise ValueError("samples must not be empty")

    variance = np.var(samples, axis=0)
    gamma = 1.0 / (variance + eps)
    return np.clip(gamma, 0.0, gamma_max)


class PrecisionEstimator:
    """Imperative Shell: окно наблюдений входа → γ.

    Хранит последние ``window`` наблюдений u(t), по ним оценивает
    обратную дисперсию каждого канала. До накопления окна использует
    имеющиеся сэмплы.

    Attributes:
        dim: Размерность канала.
        window: Максимум хранимых наблюдений.
        eps: Регуляризация (прокидывается в inverse_variance).
        gamma_max: Верхняя граница γ.
    """

    def __init__(
        self,
        dim: int,
        window: int = 50,
        eps: float = 1e-6,
        gamma_max: float = 10.0,
    ) -> None:
        """Создать оценщик точности.

        Args:
            dim: Размерность входного вектора.
            window: Размер скользящего окна (в тиках).
            eps: Регуляризация знаменателя.
            gamma_max: Верхняя граница γ (дефолт 10.0, см. inverse_variance).

        Raises:
            ValueError: Если dim <= 0 или window < 1.
        """
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")

        self.dim = dim
        self.window = window
        self.eps = eps
        self.gamma_max = gamma_max
        self._buffer: deque[Vector] = deque(maxlen=window)

    def update(self, u: Vector) -> Vector:
        """Добавить наблюдение u(t) и вернуть γ.

        Args:
            u: Вход shape=(dim,).

        Returns:
            γ shape=(dim,).

        Raises:
            ValueError: Если u.shape != (dim,).
        """
        if u.shape != (self.dim,):
            raise ValueError(f"u shape {u.shape} != (dim,)=({self.dim},)")
        self._buffer.append(np.array(u, dtype=np.float64, copy=True))
        samples = np.stack(self._buffer)
        return inverse_variance(samples, self.eps, self.gamma_max)

    @property
    def count(self) -> int:
        """Число накопленных наблюдений (до window)."""
        return len(self._buffer)

    def reset(self) -> None:
        """Очистить окно наблюдений."""
        self._buffer.clear()
