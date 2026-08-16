"""Host orchestration layer.

Temporary wiring module that connects independently developed modules
(energy, telemetry, etc.) into a working host pipeline.

This is the single location that knows about concrete field names
of FreeEnergyResult — isolating the risk of interface desync
to one place.
"""
