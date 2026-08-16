"""
Canonical Microcircuits (CMC) Fabric

Канонические колоночные микроконтуры:
- L4: входной слой (сенсорные данные)
- L5/6: генератор состояния x(t)
- L2/3: ошибка предсказания e(t)

Параллельный батчинг колонок, O(N) масштабирование.
"""
from .cmc import Column
from .energy import FreeEnergyCalculator
from .voting import KWTA

__all__ = ["Column", "FreeEnergyCalculator", "KWTA"]
