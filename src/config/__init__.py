"""Configuration module — tunable host parameters (CONSTITUTION §2.2).

Re-exports:
    HostConfig — full host loop configuration
    EnergyConfig — FreeEnergyCalculator parameters
    ColumnParams — single column parameters
    AttractorConfig — TaskAttractor parameters
    MemoryConfig — episodic memory + embedder parameters
"""

from .params import (
    AttractorConfig,
    ColumnParams,
    EnergyConfig,
    HostConfig,
    MemoryConfig,
)

__all__ = [
    "AttractorConfig",
    "ColumnParams",
    "EnergyConfig",
    "HostConfig",
    "MemoryConfig",
]
