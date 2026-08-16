"""Lateral Voting (k-WTA) — consensus between columns.

Top-k selection via stable argsort (Phase 1). Recurrent lateral inhibition
dynamics — Phase 2. O(N) argpartition — Phase 3.
"""

from .kwta import kwta
from .manager import VotingManager
from .models import VotingResult

__all__ = ["VotingManager", "VotingResult", "kwta"]
