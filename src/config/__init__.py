"""Configuration module — tunable host parameters (CONSTITUTION §2.2).

Re-exports:
    HostConfig — full host loop configuration
    EnergyConfig — FreeEnergyCalculator parameters
    ColumnParams — single column parameters
    AttractorConfig — TaskAttractor parameters
    MemoryConfig — episodic memory + embedder parameters
    SpeechConfig — speech + LLM parameters
    HomeostasisConfig — interoceptive setpoints + throttle (S4)
    PolicyConfig — action selection parameters (S4)
"""

from .params import (
    AttractorConfig,
    ColumnParams,
    EnergyConfig,
    HomeostasisConfig,
    HostConfig,
    MemoryConfig,
    PolicyConfig,
    SpeechConfig,
)

__all__ = [
    "AttractorConfig",
    "ColumnParams",
    "EnergyConfig",
    "HomeostasisConfig",
    "HostConfig",
    "MemoryConfig",
    "PolicyConfig",
    "SpeechConfig",
]
