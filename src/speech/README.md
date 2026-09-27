# Speech

Речевой актюатор (S3). Превращает состояние хоста в реплику; **не рассуждает**
(ADR-0007: LLM — зона Брока, не центр принятия решений).

Трек (манифест §4, ADR-0003):
- **Б1 (S3, реализовано):** Intent-Frame → облачный LLM API
- Б2: Activation Steering на локальной SLM (Фаза 2+)
- Б3–Б4: CMLA / Emergent Communication (Фаза 3–4)

## Состав

| Файл | Роль |
|---|---|
| `intent.py` | IntentFrame, речевые режимы (register), рендер промпта (Core) |
| `history.py` | `ConversationHistory` — буфер диалога |
| `llm.py` | LLM-клиенты (fake/api) + фабрика (env `LLM_*`) |
| `controller.py` | `should_speak` + `SpeechController` (recall → frame → LLM) |
| `chat.py` | `ChatSession` — CLI-стенд (`/clear`, `/quit`) |

## Запуск

```bash
uv run python -m src --chat --llm auto --db host_memory.db
```

`--llm auto` — реальная модель при наличии ключа, иначе fake.
`--register brief|terse|normal|story` — длина ответа.
`--reasoning` — включить reasoning (по умолчанию выкл).

Ключ: `LLM_API_KEY`, при отсутствии — универсальный `EMBEDDER_API_KEY`.

Детали — [SPEC.md](SPEC.md), [PLAN.md](PLAN.md); стадия — `stages/S3_SPEC.md`;
решения — ADR-0003 (трек речи), ADR-0007 (актюатор).