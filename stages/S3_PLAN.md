# PLAN.md — S3: Голос

Реализация `stages/S3_SPEC.md`. Порядок — от чистого ядра к обвязке, каждый
шаг с тестами. Модульные SPEC (`src/speech/SPEC.md`, `src/host/SPEC.md`,
`src/config/SPEC.md`, `src/telemetry/SPEC.md`) обновляются по ходу.

## Файлы

### Новые

1. `src/speech/intent.py` — `IntentFrame`, `REGISTER_MAX_TOKENS`,
   `describe_affect`, `build_intent_frame`, `render_messages` (Core)
2. `src/speech/llm.py` — `LlmClient` (Protocol), `LlmError`, `FakeLlmClient`,
   `ApiLlmClient`, `build_llm_client`, `llm_settings_from_env`
3. `src/speech/history.py` — `ConversationHistory` (кольцевой буфер)
4. `src/speech/controller.py` — `SpeechDecision`, `should_speak`,
   `SpeechController`
5. `src/speech/chat.py` — `ChatSession` (CLI-стенд, команды `/clear`, `/quit`)
6. `src/speech/SPEC.md`, `src/speech/PLAN.md` — модульные (после стадии)
7. `src/tests/test_speech_intent.py`
8. `src/tests/test_speech_llm.py`
9. `src/tests/test_speech_history.py`
10. `src/tests/test_speech_controller.py`
11. `src/tests/test_speech_chat.py`
12. `stages/S3_SPEC.md`, `stages/S3_PLAN.md` — эти файлы

### Изменяемые

1. `src/speech/__init__.py` — re-exports
2. `src/speech/README.md` — статус S3
3. `src/config/params.py` — `SpeechConfig` + поле `speech`; валидация
4. `src/config/__init__.py`, `src/config/SPEC.md` — re-export/доки
5. `src/telemetry/models.py` + `logger.py` — поле `spoke` (19 полей)
6. `src/telemetry/SPEC.md`
7. `src/memory/events.py` — `is_significant_event(..., has_new_message=False)`
8. `src/host/loop.py` — прокинуть «новое сообщение» в детектор значимости
9. `src/__main__.py` — `--chat` режим, `--llm {auto,fake,api}`, `--model`
10. `.env.example` — `LLM_*`
11. `src/tests/test_config.py`, `test_telemetry_*`, `test_memory_events.py` —
    адаптация + новые случаи

## Зависимости

- **Внешние:** `openai` (уже в pyproject); stdlib `logging`, `os`, `dataclasses`.
- **Внутренние:** speech ← memory (recall/embedder) + config; host ← speech (чат);
  config ← speech.

## Порядок реализации

### Шаг 1. Intent-Frame + registers (Core)

1. `IntentFrame` (frozen); `REGISTER_MAX_TOKENS` (brief/terse/normal/story).
2. `describe_affect(valence, stress) -> str`.
3. `build_intent_frame(..., register="brief")`.
4. `render_messages(frame, user_text, history=())` — system (с требованием
   краткости по register) + история + текущее сообщение + прецеденты.
5. Тесты: frame из состояния; affect по знаку/величине; register → max_tokens;
   messages структура (system+history+user); детерминизм.

### Шаг 1a. История диалога

1. `ConversationHistory(max_turns=20)`: `add_user`/`add_assistant`/`clear`/
   `as_messages`/`__len__`; кольцевой буфер.
2. Тесты: лимит глубины; порядок ролей; clear; пустая история.

### Шаг 2. LLM-клиент (Shell + Protocol)

1. `LlmClient` Protocol; `LlmError`.
2. `FakeLlmClient`: шаблонный краткий ответ (из messages, стабильный).
3. `ApiLlmClient`: ленивый OpenAI-совместимый клиент; `LlmError` при сбое;
   `max_tokens` per-call.
4. `build_llm_client(mode, model, base_url, api_key, temperature)`;
   `llm_settings_from_env()`.
5. Тесты: fake детерминирован; api без ключа → ValueError; env-настройки;
   нормализация ошибок (инъекция фейкового клиента, без сети).

### Шаг 3. SpeechController (Shell)

1. `should_speak(new_message, f, f_threshold) -> SpeechDecision`.
2. `SpeechController.recall_precedents(user_text)`.
3. `SpeechController.respond(...)`: frame → messages (с history) → LLM;
   `max_tokens` по register; сбой → None.
4. Тесты: speak по сообщению/порогу; прецеденты из recall; сбой LLM → None;
   frame содержит аффект/задачу/register; история попадает в messages.

### Шаг 4. Значимость сообщения

1. `is_significant_event(..., has_new_message=False)`.
2. Loop: новое сообщение → значимо (эпизод). Реплики хоста не пишутся.
3. Тесты: сообщение → store; без сообщения — как раньше.

### Шаг 5. Телеметрия (`spoke`)

1. +`spoke: bool` (19 полей); `log(...)` параметр с дефолтом.
2. Тесты: сериализация 19 полей; дефолт.

### Шаг 6. Config

1. `SpeechConfig` (enabled, llm_mode, f_threshold, recall_limit,
   default_register, history_turns, temperature, style); валидация.
2. `HostConfig.speech: SpeechConfig`.
3. Тесты: дефолты; валидация.

### Шаг 7. Диалоговый стенд (CLI)

1. `ChatSession(loop, controller, history, input_fn, output_fn)`; команды
   `/clear`, `/quit`.
2. `--chat`, `--llm`, `--model` в `__main__.py`; сборка controller + history.
3. Тесты: сессия с fake input/output; ответ печатается; `/clear`; `/quit`.

### Шаг 8. Документация + регресс

1. Обновить SPEC/README/SPECS/BACKLOG/VALIDATION/BUILD_ROADMAP.
2. `.env.example` — `LLM_*`.
3. `behavioral_fingerprint` — при необходимости `spoke` (пока нет).

## План тестов

| Тест | Покрытие | Инвариант |
|---|---|---|
| `test_intent_frame_from_state` | frame | поля из состояния |
| `test_describe_affect_*` | affect | знак/величина |
| `test_register_max_tokens` | register | лимиты длины |
| `test_render_messages_structure` | промпт | system+history+user |
| `test_history_*` | история | буфер, clear, роли |
| `test_fake_llm_deterministic` | LLM fake | replay |
| `test_build_llm_*` | фабрика | auto/fake/api |
| `test_llm_settings_from_env` | env | дефолты/переопределение |
| `test_should_speak_*` | решение | сообщение/порог |
| `test_controller_recall_precedents` | прецеденты | recall |
| `test_controller_respond` | ответ | frame→LLM, register |
| `test_controller_swallows_errors` | crash-safety | loop жив |
| `test_significant_on_new_message` | значимость | эпизод |
| `test_telemetry_spoke_field` | 19 полей | сериализация |
| `test_config_speech_*` | валидация | границы |
| `test_chat_session_*` | стенд | ввод→ответ, /clear |

## Заметки

- **Речь ≠ тик:** LLM вызывается из `ChatSession`, а не из `step_once` —
  медленный ответ не блокирует аффективный контур.
- **Событийность:** ответ только на новое сообщение (или F > порога).
- **Адекватность речи:** register (brief по умолчанию) + жёсткий max_tokens;
  system prompt требует краткости. Выбор register — S4.
- **История:** `ConversationHistory` в сессии, `/clear` очищает.
- **Реплики хоста не пишутся** в память (S3); критерий — S5/S6.
- **fake по умолчанию:** тесты и offline-стенд без сети/ключей.
- **Ошибки не фатальны:** LLM/recall → None + лог.
- **Env:** `LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL`; `load_dotenv()` уже в CLI.
- **Правки запоминаются:** новое сообщение → эпизод (S2 store) → recall.
- **Не путать с policy:** «ответить/промолчать по EFE» — S4; в S3 ответ на
  сообщение безусловен (если `should_speak`).
- **Reference:** `embedder.py` (env+auto+fake) — прямой образец для `llm.py`.