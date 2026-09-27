# SPEC.md — S3: Голос

Стадия S3 из `BUILD_ROADMAP.md`. Цель — хост **отвечает** в диалоге: собирает
структурированный Intent-Frame из своего состояния и прецедентов, вызывает
LLM **событийно** (не на каждый тик) и запоминает правки оператора.

Предпосылка: S1 (честные сигналы) и S2 (память + событийный коммуникативный
вход) завершены. Ворота S3 — в `VALIDATION.md` §4. Решения — ADR-0003
(Cloud LLM + Intent-Frame на Фазе 1), ADR-0004 (FC/IS), ADR-0006
(непрерывность: речь event-triggered).

## Область (входит)

1. **Intent-Frame** — структурированное представление реплики из состояния
   (F, valence, stress, активная задача, прецеденты, стиль).
2. **Рендер промпта** — IntentFrame → сообщения chat-API (Core).
3. **LLM-клиент** — OpenAI-совместимый (RouterAI) + детерминированный fake;
   настройки в `.env` (`LLM_*`).
4. **SpeechController** — событийный: решает, отвечать ли, собирает frame,
   вызывает LLM, возвращает реплику.
5. **Значимость сообщения** — новое сообщение оператора = значимое событие
   (эпизод), чтобы правки запоминались и всплывали через recall.
6. **Диалоговый стенд** — CLI-чат поверх loop (тики идут, речь — событие).

## Явно НЕ входит

- **Activation Steering / локальная SLM** — Б2 (позже).
- **CMLA / Emergent Communication** — Б3–Б4.
- **Policy / выбор действия** (ответить/промолчать/инициировать по EFE) — S4.
- **ToM, Vigilance Gate, модель партнёра** — S5.
- **EvolvingSteeringMemory** (вектор характера) — S5+; в S3 стиль — дефолт.
- **Инициатива (хост сам начинает разговор)** — заготовка по F-порогу, но
  полноценно — S4 (policy) / S6 (эпистемический драйв).

## 1. Intent-Frame (Core)

```python
@dataclass(frozen=True)
class IntentFrame:
    """Структурированное представление реплики хоста."""
    goal: str                      # цель реплики ("respond", ...)
    affect: str                    # текстовое описание аффекта
    valence: float
    stress: float
    free_energy: float
    task: str                      # активная задача/аттрактор (тег колонки)
    precedents: tuple[str, ...]    # content релевантных эпизодов из памяти
    register: str                  # речевой режим: brief/terse/normal/story
    style: str                     # стиль (S3: дефолт; S5: из характера)
```

```python
def describe_affect(valence: float, stress: float) -> str:
    """Текстовое описание аффекта по знаку/величине (детерминировано)."""

def build_intent_frame(
    *, f: float, valence: float, stress: float, task: str,
    precedents: tuple[str, ...] = (), goal: str = "respond",
    register: str = "brief", style: str = "neutral",
) -> IntentFrame: ...

def render_messages(
    frame: IntentFrame,
    user_text: str,
    history: Sequence[dict] = (),
) -> list[dict]:
    """IntentFrame + история + реплика → chat messages (system+...)."""
```

`describe_affect` — грубая детерминированная вербализация (напр. «напряжён,
негатив», «спокоен»). Полная калибровка тона — HITL на воротах.

## 1a. Речевые режимы (адекватность речи)

LLM по умолчанию склонна к «двум-трём абзацам» — так человек не говорит.
Речевой режим (`register`) задаётся в frame и управляет длиной/формой:

| Register | Форма | `max_tokens` |
|---|---|---|
| `brief` (дефолт) | 1–2 предложения | 80 |
| `terse` | одно слово / междометие («ага», «м-м») | 16 |
| `normal` | короткий абзац | 160 |
| `story` | длинный рассказ (редко) | 500 |

- System prompt **явно** требует краткости и живого диалога, запрещает абзацы
  в `brief`/`terse`.
- `max_tokens` берётся из register (жёсткий лимит), не из общего параметра.
- **Выбор** register и молчание — **S4 (policy)**; в S3 — дефолт `brief`
  с возможностью явно задать register.

## 1b. История диалога

```python
class ConversationHistory:
    """Кольцевой буфер реплик диалога (Shell, in-memory)."""
    def __init__(self, max_turns: int = 20) -> None: ...
    def add_user(self, text: str) -> None: ...
    def add_assistant(self, text: str) -> None: ...
    def clear(self) -> None: ...
    def as_messages(self) -> list[dict]: ...   # [{role, content}, ...]
    def __len__(self) -> int: ...
```

- Хранится в сессии (in-memory; персистентность диалога — позже).
- Ограничена `max_turns` (последние N реплик) — контекст не растёт бесконечно.
- Команда `/clear` в чате очищает историю (и, опционально, память эпизодов —
  см. §5).

## 2. LLM-клиент (Shell, DI через Protocol)

```python
class LlmClient(Protocol):
    def reply(self, messages: list[dict], max_tokens: int = 256) -> str: ...

class LlmError(Exception): ...

class FakeLlmClient:
    """Детерминированный: шаблонный ответ по register-подсказке (тесты/replay)."""
    def reply(self, messages: list[dict], max_tokens: int = 256) -> str: ...

class ApiLlmClient:
    """OpenAI-совместимый (RouterAI), ленивый клиент, retry-free."""
    def __init__(self, model: str = ..., base_url: str = ..., api_key: str | None = None,
                 temperature: float = 0.7, reasoning: bool = False) -> None: ...
    def reply(self, messages: list[dict], max_tokens: int = 256) -> str: ...
```

### Конфигурация через окружение (.env)

| Переменная | Дефолт | Смысл |
|---|---|---|
| `LLM_API_KEY` | — | Ключ (опционально; иначе универсальный) |
| `EMBEDDER_API_KEY` | — | **Универсальный ключ** RouterAI (fallback для LLM) |
| `LLM_BASE_URL` | `https://routerai.ru/api/v1` | OpenAI-совместимый base URL |
| `LLM_MODEL` | `deepseek/deepseek-v4.1-flash` | Модель чата |

RouterAI использует **один ключ на все модели**: если `LLM_API_KEY` не задан,
клиент берёт `EMBEDDER_API_KEY` (универсальный). Явный `LLM_API_KEY` важнее.

Режим `llm_mode`: `auto` (ключ→api, иначе fake) / `fake` / `api`.
`build_llm_client(mode, ...)` + `llm_settings_from_env()`.

`deepseek/deepseek-v4.1-flash` (RouterAI): 4,22 ₽ вход / 34 ₽ выход, 223 ток/с,
отклик ~0,8 с, контекст 1M, поддерживает `seed` (детерминизм) и стриминг.

### Reasoning отключён (ADR-0007)

LLM здесь — **речевой актюатор (зона Брока)**, не центр рассуждения.
`ApiLlmClient` по умолчанию вызывает chat с `reasoning: {enabled: false}`.
Без этого reasoning-модель тратит токены на «размышление» и возвращает
**пустой `content`** (`finish_reason: length`), а также сильнее
переинтерпретирует Intent-Frame. Ручка `reasoning` (+ CLI `--reasoning`)
включит режим, когда ядро (S4/S6) начнёт формировать развёрнутые намерения.

**Ключевое отличие от эмбеддера:** реальный ответ **недетерминирован**
(temperature), поэтому тесты проверяют **структуру** (Intent-Frame, вызов,
лимиты), а не текст. `FakeLlmClient` даёт стабильный шаблон для тестов.

## 3. SpeechController (Shell)

```python
@dataclass(frozen=True)
class SpeechDecision:
    speak: bool
    reason: str

def should_speak(*, new_message: bool, f: float, f_threshold: float) -> SpeechDecision:
    """Отвечать, если есть новое сообщение ИЛИ F > порога (инициатива)."""

class SpeechController:
    def __init__(self, llm: LlmClient, memory: SupportsRecall | None,
                 embedder: Embedder | None, f_threshold: float,
                 recall_limit: int = 3, default_register: str = "brief") -> None: ...

    def recall_precedents(self, user_text: str) -> tuple[str, ...]: ...

    def respond(
        self, *, user_text: str, f: float, valence: float, stress: float,
        task: str, history: Sequence[dict] = (), register: str | None = None,
    ) -> str | None:
        """Событийный ответ: frame → messages (с историей) → LLM.

        None, если не отвечаем. Ошибки LLM/recall → None (речь не роняет тик).
        """
```

`max_tokens` выбирается по `register` (см. §1a), а не общим параметром.

## 4. Значимость коммуникативного входа

Новое сообщение оператора — **всегда значимое событие** (коммуникативный
вход событиен по природе, ADR-0006). Это расширяет детектор S2:

```python
def is_significant_event(f, prev_f, reflex_tags, spike_threshold,
                         has_new_message: bool = False) -> bool:
    """reflex ИЛИ всплеск F ИЛИ новое сообщение."""
```

Следствие: каждое сообщение оператора → эпизод (`content` = текст) →
recall возвращает его при похожем запросе → правки «запоминаются» и влияют
на следующий ответ (ворота S3).

## 5. Диалоговый стенд (CLI)

Отдельный режим `python -m src --chat`: цикл диалога поверх loop.

```
loop идёт тиками (аффективный контур)
ввод оператора → history.add_user + message в loop → N тиков («осмыслено»)
              → SpeechController.respond(..., history) → печать
              → history.add_assistant
```

- Ввод → `messages` провайдера (loop эмбеддит при новом тексте — событийно).
- Ответ печатается и добавляется в историю.
- **Команды:** `/clear` (очистить историю диалога), `/quit` (выход).
- Реализация — Shell (`ChatSession`), тестируется через инъекцию
  `input_fn`/`output_fn`.

```python
class ChatSession:
    def __init__(self, loop: HostLoop, controller: SpeechController,
                 history: ConversationHistory | None = None,
                 input_fn=input, output_fn=print) -> None: ...
    def run(self, max_turns: int = 0) -> int: ...   # 0 → до /quit
```

## 5a. Что записывается в память

| Источник | Записывать? | Обоснование |
|---|---|---|
| Сообщение оператора | **Всегда** | Коммуникативный вход событиен (S2); правки должны запоминаться |
| Реплика хоста | **Нет** (S3) | Иначе память — лента болтовни, recall вернёт шум |

Критерий «выдающейся» реплики хоста (что стоит помнить) — открытый вопрос,
связан с характером/биографией (S5/S6). В S3 реплики хоста не пишутся.

## 6. Конфигурация

Новая `SpeechConfig` (+ поле `speech` в `HostConfig`):

| Поле | Дефолт | Смысл |
|---|---|---|
| `enabled` | False | Включать речь (S3 по умолчанию выкл; включается `--chat`) |
| `llm_mode` | "auto" | auto (ключ→api, иначе fake), fake, api |
| `f_threshold` | 1.0 | Порог F для инициативы (не для ответа на сообщение) |
| `recall_limit` | 3 | Сколько прецедентов подавать в frame |
| `default_register` | "brief" | Речевой режим по умолчанию |
| `history_turns` | 20 | Глубина истории диалога (реплик) |
| `temperature` | 0.7 | Температура генерации |
| `style` | "neutral" | Дефолтный стиль (S5 — из характера) |
| `reasoning` | False | Включить reasoning у LLM (ADR-0007) |

Валидация: `f_threshold ≥ 0`, `recall_limit ≥ 1`, `history_turns ≥ 0`,
`temperature ∈ [0, 2]`, `llm_mode ∈ {auto, fake, api}`,
`default_register ∈ {brief, terse, normal, story}`.

Лимиты `max_tokens` по register — в `REGISTER_MAX_TOKENS` (Core-константа).

## 7. Телеметрия (минимальное расширение)

| Поле | Тип | Смысл |
|---|---|---|
| `spoke` | bool | Хост сгенерировал реплику на тике |

Обосновано наблюдаемостью (VALIDATION §2.5). Итого 19 полей.

## Инварианты S3

1. **Речь не роняет контур:** ошибки LLM/recall → `None`/лог, loop живёт.
2. **Event-triggered:** LLM вызывается не чаще, чем приходит сообщение
   (или F > порога для инициативы), не на каждый тик.
3. **Frame детерминирован:** `build_intent_frame`/`render_messages` — чистые.
4. **Тон следует аффекту:** `affect`/`valence` в frame соответствуют
   состоянию (проверяется структурно + HITL).
5. **Правки запоминаются:** новое сообщение → эпизод → recall находит.
6. **Обратная совместимость:** `speech.enabled=False` → контур S1/S2.

## Критерии приёмки

См. `VALIDATION.md` §4 (ворота S3). Кратко:
- [ ] Intent-Frame собирается из состояния (F, valence, аттрактор, прецеденты)
- [ ] вызов LLM только при новом сообщении / F > порога (экологическая рациональность)
- [ ] тон ответа следует знаку/величине valence (судит человек, HITL)
- [ ] правка оператора запоминается и влияет на следующий ответ
- [ ] `llm_mode="fake"` → детерминированный стенд для тестов
- [ ] `ruff`/тесты зелёные; mypy strict для `src/speech/`

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Модель LLM по умолчанию | Решено | `deepseek/deepseek-v4.1-flash` (RouterAI) |
| Триггер ответа | Предложено | Новое сообщение → всегда; F-порог → инициатива (S4) |
| Новое сообщение = эпизод | Предложено | Да (коммуникативный вход событиен) |
| Адекватность речи (длина) | Решено | Речевые режимы (register) + жёсткий max_tokens; выбор — S4 |
| История диалога | Решено | `ConversationHistory` + `/clear`; глубина `history_turns` |
| Реплики хоста в память | Решено (S3) | Не писать; критерий «выдающегося» — S5/S6 |
| Стиль | Отложено | Дефолт в S3; из характера — S5 |
| История в промпте | Решено | Да: system + история + текущее + прецеденты |
| Стриминг ответа | Отложено | S3: обычный вызов; стриминг — позже |
| Retry/timeout LLM | Отложено | S3: без retry; timeout — параметр позже |
| Reasoning у LLM | Решено | Отключён (ADR-0007: LLM — актюатор, не рассуждающий) |

## Implementation Notes

1. **FC/IS:** `intent.py` (Core) — чистый; `llm.py`/`controller.py`/`chat.py` —
   Shell с DI.
2. **Событийность:** контроллер вызывается диалоговым стендом (не loop'ом),
   чтобы медленный LLM не блокировал тики.
3. **Речь ≠ тик:** loop продолжает идти; реплика — отдельное событие поверх.
4. **`--chat` включает речь:** `speech.enabled` дефолт False, CLI поднимает.
5. **Прецеденты:** recall по эмбеддингу сообщения → `content` top-k.
6. **Env:** `LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL`; `LLM_API_KEY` опционален
   (fallback — универсальный `EMBEDDER_API_KEY`); `load_dotenv()` в CLI.
7. **Reasoning выключен:** `ApiLlmClient(reasoning=False)` по умолчанию
   (ADR-0007); включается `--reasoning`/`SpeechConfig.reasoning`.
8. **Reference:** `Embedder`/`build_embedder` (S2) — тот же паттерн env+auto+fake;
   `TelemetryLogger` (DI через Protocol).