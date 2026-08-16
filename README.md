# neuro2

Адаптивный хост на основе принципов неокортекса: канонические колоночные микроконтуры, свободная энергия, активное выведение.

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
│   └── voting/     # k-WTA lateral inhibition
├── memory/         # SQLite + sqlite-vec
├── speech/         # Intent-Frame, Steering
├── mcp/            # MCP Integration
├── tm/             # Theory of Mind
├── telemetry/      # Логирование, самодиагностика
└── config/         # Конфигурация
```

## Запуск

```bash
uv run python -m src
```

## Лицензия

Private
