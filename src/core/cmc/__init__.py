"""
Canonical Microcircuit Column

Layers: L4 -> L5/6 -> L2/3
Specialization: tone, rhythm, meanings, ToM, MCP-sensors
"""
import numpy as np


class Column:
    """Каноническая колонка CMC."""

    def __init__(self, input_dim: int, state_dim: int, specialization: str = "general") -> None:
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.specialization = specialization
        self.x = np.zeros(state_dim)  # состояние L5/6
        self.e = np.zeros(input_dim)  # ошибка L2/3

    def forward(self, u: np.ndarray) -> np.ndarray:
        """L4 -> L5/6 -> L2/3."""
        self.x = self._state_update(u)
        self.e = self._prediction_error(u, self.x)
        return self.e

    def _state_update(self, u: np.ndarray) -> np.ndarray:
        """Обновление состояния."""
        return u

    def _prediction_error(self, u: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Ошибка предсказания."""
        return u - x
