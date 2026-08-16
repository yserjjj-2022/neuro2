"""
Canonical Microcircuit Column

Слои: L4 → L5/6 → L2/3
Специализация: тон, ритм, смыслы, ToM, MCP-сенсоры
"""
import numpy as np

class Column:
    """Каноническая колонка CMC."""
    
    def __init__(self, input_dim: int, state_dim: int, specialization: str = "general"):
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.specialization = specialization
        self.x = np.zeros(state_dim)  # состояние L5/6
        self.e = np.zeros(input_dim)  # ошибка L2/3
        
    def forward(self, u: np.ndarray) -> np.ndarray:
        """L4 → L5/6 → L2/3."""
        # L4 (вход) → L5/6 (состояние)
        self.x = self._state_update(u)
        # L2/3 (ошибка предсказания)
        self.e = self._prediction_error(u, self.x)
        return self.e
    
    def _state_update(self, u: np.ndarray) -> np.ndarray:
        """Обновлени"""
Canonical Microcircuit Column

С?rCaur
Слои: L4 → L5/6 → Lп??пециализация: ??"""
import numpy as np

class Column:
    """Каноническая колонк nimnd
class Column:
  "О    """Кап?   
    def __init__(self, input_dim: int, state_dim               self.input_dim = input_dim
        self.state_dim = state_dim
        self.spp.ze        self.state_dim = state_die = np.zeros(self.input_dim)
