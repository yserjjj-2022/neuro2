"""Telemetry module — host state logging for calibration and analysis.

Re-exports:
    TelemetryEvent — flat serializable dataclass
    serialize_event — pure functional core
    TelemetryWriter — imperative shell (file I/O)
    TelemetryLogger — shadow observer with DI
    SupportsWrite — Protocol for duck typing
"""

from .logger import SupportsWrite, TelemetryLogger
from .models import TelemetryEvent
from .serialize import serialize_event
from .writer import TelemetryWriter

__all__ = [
    "SupportsWrite",
    "TelemetryEvent",
    "TelemetryLogger",
    "TelemetryWriter",
    "serialize_event",
]
