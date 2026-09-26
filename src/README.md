# src

Исходный код адаптивного хоста neuro2.

```
src/
├── core/       # Колоночное ядро: CMC, energy, voting, attractors
├── host/       # Обвязка: wiring, sources (сенсорика), loop (host loop)
├── memory/     # Эпизодическая память (SQLite + sqlite-vec)
├── speech/     # Речевой актюатор (не реализовано)
├── mcp/        # Сенсорика/действия MCP (пока только контракт сигналов)
├── tm/         # Theory of Mind (не реализовано)
├── telemetry/  # JSONL-логирование состояния
├── config/     # Параметры (пороги, dt, режимы)
└── tests/      # pytest
```

Дисциплина: CONSTITUTION.md (SDD, FC/IS, ruff, mypy strict для `core/`).