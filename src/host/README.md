# host

Обвязка хоста: связывает модули в работающую систему и оживляет её во времени.

- `wiring.py` — сборка per-tick конвейера: cmc → voting/attractors → energy
  → telemetry. Единственное место, знающее конкретные поля `FreeEnergyResult`.
- `sources.py` — сенсорная шина: провайдеры сигналов (mock/real) → `u(t)`,
  карта сегментов шины (фундамент width scaling), медленный такт.
- `loop.py` — `HostLoop`: цикл по `dt`, precision, graceful shutdown.

См. `SPEC.md` и `PLAN.md`. Точка входа: `uv run python -m src`.