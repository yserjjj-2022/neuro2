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
    SocialConfig — theory-of-mind parameters (S5)
    AutonomyConfig — selfcontrol/consolidation/drive/factors parameters (S6)
"""

from .params import (
    AttractorConfig,
    AutonomyConfig,
    ColumnParams,
    EnergyConfig,
    HomeostasisConfig,
    HostConfig,
    MemoryConfig,
    PolicyConfig,
    SocialConfig,
    SpeechConfig,
)

__all__ = [
    "AttractorConfig",
    "AutonomyConfig",
    "ColumnParams",
    "EnergyConfig",
    "HomeostasisConfig",
    "HostConfig",
    "MemoryConfig",
    "PolicyConfig",
    "SocialConfig",
    "SpeechConfig",
]
