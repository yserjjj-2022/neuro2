# neuro2

Адаптивный хост на основе принципов неокортекса: канонические колоночные микроконтуры, свободная энергия, активное выведение.

## Навигация по проекту

- [`INTENT.md`](INTENT.md) — зачем проект: выращивание, а не программирование
- [`BUILD_ROADMAP.md`](BUILD_ROADMAP.md) — порядок сборки S1–S6, ворота
- [`VALIDATION.md`](VALIDATION.md) — проверка, сценарии, инварианты
- [`SPECS.md`](SPECS.md) — реестр модульных спецификаций
- [`CONSTITUTION.md`](CONSTITUTION.md) — правила проекта
- [`host_architecture_manifest.md`](host_architecture_manifest.md) — архитектура
- [`BACKLOG.md`](BACKLOG.md) — задачи по стадиям
- [`adr/`](adr/) — Architecture Decision Records
- [`stages/`](stages/) — SPEC/PLAN по стадиям сборки

## Установка

```bash
uv sync
```

## Структура

```
src/
├── core/           # Колоночное ядро (CMC)
│   ├── cmc/        # Canonical Microcircuits
│   ├── energy/     # Free Energy, valence
│   ├── voting/     # k-WTA lateral inhibition
│   └── attractors/ # Task attractors (STP)
├── host/           # Обвязка: wiring (pipeline), sources (сенсорика)
├── memory/         # SQLite + sqlite-vec
├── speech/         # Intent-Frame, Steering (не реализовано)
├── mcp/            # Контракт сигналов; MCP transport (не реализован)
├── tm/             # Theory of Mind (не реализовано)
├── telemetry/      # Логирование, самодиагностика
└── config/         # Конфигурация
```

## Статус

Host loop собран и работает: `u(t) → CMC → voting/attractors → energy →
telemetry (JSONL)`. **S1 (честные сигналы) завершён:** единая временная база,
сглаженная valence, настоящая γ, ресурсная интероцепция, guard дрейфа;
286 тестов. Мок-сенсорика (`src/host/sources.py`), параметры (`src/config/`),
CLI. Далее — S2 (непрерывность: эмбеддер + память).

Цель и рамка — в [`INTENT.md`](INTENT.md): выращивание нейроперсоны, не
программирование поведения.

## Запуск

```bash
uv run python -m src --ticks 100 --log run.jsonl
```

`--ticks 0` — бесконечный цикл до Ctrl+C. Полный список: `--help`.

## Лицензия

Private
