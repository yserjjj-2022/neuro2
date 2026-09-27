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
cp .env.example .env   # заполнить EMBEDDER_API_KEY (RouterAI)
```

Эмбеддер по умолчанию — RouterAI (`https://routerai.ru`), настройки в `.env`
(не коммитится): `EMBEDDER_API_KEY`, `EMBEDDER_BASE_URL`, `EMBEDDER_MODEL`,
`EMBEDDER_DIM`. Без ключа `embedder_mode="auto"` использует детерминированный
fake-эмбеддер (тесты/replay).

Ключ RouterAI — **универсальный** (один на все модели): LLM берёт
`LLM_API_KEY`, а при его отсутствии — `EMBEDDER_API_KEY`. Модель чата — в
`LLM_MODEL` (дефолт `deepseek/deepseek-v4.1-flash`).

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
├── speech/         # Intent-Frame, LLM-актюатор, диалог (S3)
├── mcp/            # Контракт сигналов; MCP transport (не реализован)
├── tm/             # Theory of Mind (не реализовано)
├── telemetry/      # Логирование, самодиагностика
└── config/         # Конфигурация
```

## Статус

Host loop собран и работает: `u(t) → CMC → voting/attractors → energy →
telemetry (JSONL)`. **S1 (честные сигналы) завершён:** единая временная база,
сглаженная valence, настоящая γ, ресурсная интероцепция, guard дрейфа.
**S2 (непрерывность) завершён:** эмбеддер, эпизодическая память подключена
(recall → приор в шину, запись на значимых событиях). **S3 (голос):**
Intent-Frame + речевые режимы, event-triggered LLM (речевой актюатор,
reasoning выключен — ADR-0007), история диалога, `--chat`. 431 тест.
Мок-сенсорика (`src/host/sources.py`), параметры (`src/config/`), CLI.

Цель и рамка — в [`INTENT.md`](INTENT.md): выращивание нейроперсоны, не
программирование поведения.

## Запуск

```bash
uv run python -m src --ticks 100 --log run.jsonl --db host_memory.db
```

`--ticks 0` — бесконечный цикл до Ctrl+C. Полный список: `--help`.

Диалоговый режим (S3):

```bash
uv run python -m src --chat --llm auto --db host_memory.db
```

`--llm auto` берёт реальную модель при наличии ключа, иначе детерминированный
fake. Команды в чате: `/clear` (очистить историю), `/quit` (выход).
`--register brief|terse|normal|story` — длина ответа; `--reasoning` включает
reasoning у модели (по умолчанию выкл); `--status` печатает состояние хоста
(F, valence, stress, γ, задача, recall, дрейф) перед каждой репликой.
Подробнее — [src/speech/README.md](src/speech/README.md).

## Лицензия

Private
