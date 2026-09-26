# Telemetry

Плоское JSONL-логирование состояния хоста. S1: 15 полей — F/valence/stress/
gamma, tick, active_tags, reflex_tags, bus_dim, latency_ms, rss_mb, drift,
phase/mode.

- Core: `serialize_event` (JSON, `allow_nan=False`)
- Shell: `TelemetryWriter` (файл, flush), `TelemetryLogger` (DI, crash-safety)

В S1 writer'ом владеет host loop (у него полный контекст). См. `SPEC.md`.