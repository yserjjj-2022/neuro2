# src

Исходный код адаптивного хоста neuro2.

```
src/
├── core/       # Ядро: CMC, energy, voting, attractors, homeostasis, policy
├── host/       # Обвязка: wiring, sources, loop, throttle (S4)
├── memory/     # Эпизодическая память (SQLite + sqlite-vec)
├── speech/     # Речевой актюатор (S3, + goal из policy S4)
├── mcp/        # Сенсорика/действия MCP (пока только контракт сигналов)
├── tm/         # Theory of Mind (не реализовано)
├── telemetry/  # JSONL-логирование состояния
├── config/     # Параметры (пороги, dt, режимы)
└── tests/      # pytest
```

Дисциплина: CONSTITUTION.md (SDD, FC/IS, ruff, mypy strict для `core/`).