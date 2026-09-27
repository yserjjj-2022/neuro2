# PLAN.md — src/speech

План реализации S3 (речь). Источник — `stages/S3_PLAN.md`.

## Шаги

- [x] **1. Intent-Frame + registers + история**
  - `intent.py`: `IntentFrame`, `describe_affect`, `build_intent_frame`,
    `render_messages`, `REGISTER_MAX_TOKENS`, `register_max_tokens`.
  - `history.py`: `ConversationHistory` (кольцевой буфер, clear).
  - Тесты: `test_speech_intent.py`, `test_speech_history.py`.
- [x] **2. LLM-клиент**
  - `llm.py`: `LlmClient`, `LlmError`, `FakeLlmClient`, `ApiLlmClient`,
    `build_llm_client`, `llm_settings_from_env`, fallback на `EMBEDDER_API_KEY`.
  - Тесты: `test_speech_llm.py`.
- [x] **3. SpeechController**
  - `controller.py`: `should_speak`, `SpeechController` (recall → frame → LLM,
    crash-safety).
  - Тесты: `test_speech_controller.py`.
- [x] **4. Значимость коммуникативного входа**
  - `memory/events.py`: `is_significant_event(..., has_new_message)`.
  - `memory/router.py` + `host/loop.py`: проброс `has_new_message`.
- [x] **5. Телеметрия `spoke`** (19 полей)
  - `telemetry/models.py`, `telemetry/logger.py`, `host/loop.py`
    (`mark_spoke`).
- [x] **6. `SpeechConfig`**
  - `config/params.py` (+ поле в `HostConfig`), валидация; `config/__init__.py`.
- [x] **7. Диалоговый стенд**
  - `chat.py`: `ChatSession`; `__main__.py`: `--chat/--llm/--model/--register/
    --f-threshold/--history-turns/--reasoning`.
  - `core/cmc/ensemble.py`: свойство `column_configs`; `host/loop.py`:
    `last_outcome`.
- [x] **7a. Мини-индикатор состояния**
  - `status.py`: `format_status`; `ChatSession(show_status=...)`;
    `--status`; `host/loop.py`: `last_drift`/`last_memory_hit`.
- [x] **8. Reasoning отключён (ADR-0007)**
  - `ApiLlmClient.reasoning=False` по умолчанию + ручка.
- [x] **9. Документация**
  - `adr/0007-*`, `stages/S3_SPEC.md`, `src/speech/SPEC.md`/`PLAN.md`/`README.md`,
    `SPECS.md`, `README.md`, `BACKLOG.md`, `VALIDATION.md`, `.env.example`.

## Заметки

- **Речь ≠ тик:** LLM вызывается из `ChatSession`, не из `step_once`.
- **fake по умолчанию:** тесты и offline-стенд без сети/ключей.
- **Ключ универсальный:** `LLM_API_KEY` → fallback `EMBEDDER_API_KEY`.
- **Reasoning выключен:** LLM — актюатор, не рассуждающий (ADR-0007).
- **Реплики хоста не пишутся** в память (S3); критерий — S5/S6.
