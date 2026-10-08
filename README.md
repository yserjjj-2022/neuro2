# neuro2

Адаптивный хост на основе принципов неокортекса: канонические колоночные микроконтуры, свободная энергия, активное выведение.

## Навигация по проекту

- [`INTENT.md`](INTENT.md) — зачем проект: выращивание, а не программирование
- [`BUILD_ROADMAP.md`](BUILD_ROADMAP.md) — порядок сборки S1–S8, ворота
- [`VALIDATION.md`](VALIDATION.md) — проверка, сценарии, инварианты
- [`SPECS.md`](SPECS.md) — реестр модульных спецификаций
- [`CONSTITUTION.md`](CONSTITUTION.md) — правила проекта
- [`host_architecture_manifest.md`](host_architecture_manifest.md) — архитектура
- [`BACKLOG.md`](BACKLOG.md) — задачи по стадиям
- [`adr/`](adr/) — Architecture Decision Records
- [`stages/`](stages/) — SPEC/PLAN по стадиям сборки
  (запланированный спринт — [`stages/ACTUATION_PLAN.md`](stages/ACTUATION_PLAN.md), S8)

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
│   ├── attractors/ # Task attractors (STP)
│   ├── homeostasis/# Гомеостаз (S4)
│   ├── policy/     # Выбор действия (S4/S6)
│   ├── selfcontrol/# Метакогниция, дрейф, сброс (S6)
│   └── factorization/# Разреженный дискретный слой (S6)
├── host/           # Обвязка: wiring, sources, fingerprint/diagnostic (S7)
├── memory/         # SQLite + sqlite-vec + консолидация (S6)
├── speech/         # Intent-Frame, LLM-актюатор, диалог (S3)
├── mcp/            # Контракт сигналов + карта аффордансов + MCP-клиент (stdio)
├── integrations/   # Реестр подключений (каталог) + сборка runtime (ADR-0011)
├── tm/             # Theory of Mind (S5)
├── telemetry/      # Логирование, самодиагностика
└── config/         # Конфигурация
```

## Статус

Host loop собран и работает: `u(t) → CMC → voting/attractors → energy →
telemetry (JSONL)`. **S1 (честные сигналы) завершён:** единая временная база,
сглаженная valence, настоящая γ, ресурсная интероцепция, guard дрейфа.
**S2 (непрерывность) завершён:** эмбеддер, эпизодическая память подключена
(recall → приор в шину, запись на значимых событиях). **S3 (голос) завершён:**
Intent-Frame + речевые режимы, event-triggered LLM (речевой актюатор,
reasoning выключен — ADR-0007), история диалога, `--chat`, `--status`.
**S4 (воля) завершён:** policy, гомеостаз, reflex-throttle, γ-барьер, gate
(гранулярные права), escape hatch, grounding IntentFrame. **S5 (социальность)
завершён:** ToM (сигнатура/узнавание), тайминг диалога, Vigilance Gate,
имена, Joint Agency. **S6 (автономия) завершён:** самоконтроль (метакогниция,
critical slowing down, протокол сброса), консолидация памяти (pruning + схемы,
ночной цикл по расписанию), эпистемический драйв `Action.EXPLORE` (реальный
эффектор: карта аффордансов + gated-зондирование), NumPy-факторизация
(mode/partner/task), длинный горизонт C10. **S7 (HITL-диагностика) завершён:**
формальный sensitivity-harness (`--sensitivity`, инварианты направления), пресеты
(Python-база + TOML-override, `--preset`/`--preset-file`), диалоговая диагностика
(дерево проб S3–S6 + ветвление, `--diagnose`), самоотчёт сброса через
`CapabilityGate` (fail-safe deny); операторский протокол —
`stages/S7_HITL_PROTOCOL.md` (ADR-0010). **Реестр интеграций + MCP-транспорт
реализован** (ADR-0011): каталог подключений (Python-база + TOML-override) +
реальные MCP-серверы (stdio) вместо mock-зондирования, флаг `--integrations PATH`.
1061 тест.
Мок-сенсорика (`src/host/sources.py`), параметры (`src/config/`), CLI.

**Запланировано:** `S8` секвенирование актуаций (executive-слой, Behavior Tree,
асинхронность, MCP как действие) — [`stages/ACTUATION_PLAN.md`](stages/ACTUATION_PLAN.md),
не начато.

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

Диагностика (S7):

```bash
uv run python -m src --sensitivity                      # матрица ручек → инварианты
uv run python -m src --behavioral                       # поведенческий автотест по звеньям
uv run python -m src --preset dialogue --ticks 200      # именованный пресет
uv run python -m src --diagnose --diagnose-log diag.jsonl   # дерево проб S3–S6
```

`--preset baseline|stress|dialogue|autonomy|long-horizon|cooperative` +
`--preset-file configs/*.toml` (fail-fast override). `--diagnose` ведёт пробу →
snapshot → категориальный вердикт → ветвление; `--probes FILE` переопределяет
дерево. `--behavioral [--behavioral-precondition born|primed|matured]
[--behavioral-json PATH] [--behavioral-baseline PATH]
[--behavioral-save-baseline PATH] [--behavioral-band F]` прогоняет поведенческий
автотест (§7) и печатает сводку (в JSON — машинный отчёт); эталон наблюдаемых
сохраняется (`--behavioral-save-baseline`) и сравнивается по полосе
(`--behavioral-baseline`/`--behavioral-band`). Операторский протокол —
`stages/S7_HITL_PROTOCOL.md`.

## Лицензия

Private
