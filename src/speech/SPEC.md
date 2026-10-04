# SPEC.md — src/speech

## Назначение

Речевой актюатор хоста (аналог зоны Брока): превращает внутреннее состояние
(аффект, аттрактор, прецеденты) в реплику. **LLM не рассуждает** — она
артикулирует уже сформированное ядром намерение (ADR-0007). Трек речи —
манифест §4, ADR-0003 (Б1: Intent-Frame → облачный LLM, Фаза 1).

Стадия S3 (`stages/S3_SPEC.md`). Речь **event-triggered**: LLM вызывается не
на каждый тик, а на новое сообщение (или F > порога).

См. также:
- `Embedder`/`build_embedder` (S2) — образец паттерна env + auto/fake/api
- `TelemetryLogger` — образец Shell с DI через Protocol
- `MemoryRouter` — источник прецедентов (recall)

## Состав

| Файл | Слой | Роль |
|---|---|---|
| `intent.py` | Core | `IntentFrame`, `describe_affect`, `build_intent_frame`, `render_messages`, `REGISTER_MAX_TOKENS`, `GOAL_INSTRUCTIONS`/`goal_instruction`, `escape_hatch_message` |
| `history.py` | Shell | `ConversationHistory` — кольцевой буфер диалога |
| `llm.py` | Shell | `LlmClient` (Protocol), `FakeLlmClient`, `ApiLlmClient`, `build_llm_client`, `llm_settings_from_env` |
| `controller.py` | Shell | `should_speak`, `SpeechController` |
| `status.py` | Core | `format_status` — строка состояния для стенда |
| `chat.py` | Shell | `ChatSession` — CLI-стенд (`/clear`, `/quit`) |

## Core: Intent-Frame и речевые режимы

```python
@dataclass(frozen=True)
class IntentFrame:
    goal: str
    affect: str
    valence: float
    stress: float
    free_energy: float
    task: str
    precedents: tuple[str, ...]
    register: str = "brief"
    style: str = "neutral"


def describe_affect(valence: float, stress: float) -> str: ...


def build_intent_frame(
    *,
    f,
    valence,
    stress,
    task,
    precedents=(),
    goal="respond",
    register="brief",
    style="neutral",
) -> IntentFrame: ...


def render_messages(frame, user_text, history=()) -> list[dict]: ...


def register_max_tokens(register: str) -> int: ...


def goal_instruction(goal: str) -> str: ...


def escape_hatch_message(*, task="none", stress=0.0) -> str: ...
```

`REGISTER_MAX_TOKENS`: `terse`=16, `brief`=80 (дефолт), `normal`=160,
`story`=500. System prompt требует краткости и живого диалога; жёсткий
`max_tokens` ограничивает «два-три абзаца». Выбор register — S4 (policy).

**Grounding goal → промпт** (S4-долг): `GOAL_INSTRUCTIONS` сопоставляет
низкоуровневой цели policy (`respond`/`initiative`/`identify_partner`/`silent`)
явную инструкцию + scope. `goal_instruction(goal)` попадает в system-промпт;
неизвестная цель → безопасный дефолт `respond`. Без этого LLM трактовала
`goal` семантически широко (аудит S4: семантический дрейф).

**Escape hatch** (S4-долг): `escape_hatch_message` — детерминированная дешёвая
реплика о перегрузке (без LLM) для случая удержанного throttle. Используется
`ChatSession`, когда `loop.escape_hatch_active`.

## Shell: LLM-клиент

```python
class LlmClient(Protocol):
    def reply(self, messages: list[dict], max_tokens: int = 256) -> str: ...

class LlmError(Exception): ...

class FakeLlmClient:      # детерминированный шаблон по register-подсказке
class ApiLlmClient:       # OpenAI-совместимый (RouterAI), ленивый, retry-free

def build_llm_client(mode="auto", model=..., base_url=..., api_key=None,
                     temperature=0.7, reasoning=False) -> LlmClient: ...
def llm_settings_from_env() -> dict[str, str]: ...
```

- Настройки в `.env`: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.
- Ключ **опционален**: при отсутствии `LLM_API_KEY` берётся универсальный
  `EMBEDDER_API_KEY` (RouterAI — один ключ на все модели).
- **Reasoning отключён** по умолчанию (`reasoning: {enabled: false}`): иначе
  reasoning-модель возвращает пустой `content`. Ручка `reasoning`/`--reasoning`
  — на будущее (S4/S6). ADR-0007.

## Shell: SpeechController

```python
@dataclass(frozen=True)
class SpeechDecision:
    speak: bool
    reason: str


def should_speak(*, new_message, f, f_threshold) -> SpeechDecision: ...


class SpeechController:
    def __init__(self, llm, memory=None, embedder=None, f_threshold=1.0,
                 recall_limit=3, default_register="brief") -> None: ...
    def recall_precedents(self, user_text: str) -> tuple[str, ...]: ...
    def respond(self, *, user_text, f, valence, stress, task, history=(),
                register=None, new_message=True, goal=None) -> str | None: ...
```

Новое сообщение → всегда отвечаем; иначе — инициатива при F > порога.
Ошибки LLM/recall → `None` + лог (речь не роняет тик).

`goal` (S4) — цель реплики из policy (`respond`/`initiative`/`identify_partner`);
`None` → `"respond"` (S3-совместимость). `ChatSession` при включённой policy
вызывает `select_action` и передаёт цель; `SILENT` → `[хост промолчал]`.
Рефлекс-throttle (`llm_gate`) блокирует инициативу, но не ответ на сообщение.
Под удержанным throttle (`escape_hatch_active`) `ChatSession` отвечает
шаблоном `escape_hatch_message` без LLM — «право подать голос» сохранено
(гранулярный gate: `SPEAK` выдан, `THINK` отозван).

## Shell: ChatSession

```python
class ChatSession:
    def __init__(
        self,
        loop,
        controller,
        history=None,
        input_fn=input,
        output_fn=print,
        ticks_per_turn=3,
        show_status=False,
        policy=None,
        gate=None,
        partner_model=None,      # S5: ToM; None → S4-совместимость
        pause_tau_s=5.0,
        clock=time.monotonic,
        vigilance=None,          # S5: Vigilance Gate
        joint_agency=None,       # S5: Joint Agency
    ) -> None: ...
    def run(self, max_turns: int = 0) -> int: ...
```

Ввод → сообщение в loop → N тиков → `controller.respond` → печать.

**ToM (S5):** `ChatSession` владеет `PartnerModel`; на реплике считает
нормированную паузу (`tm.normalize_pause`) и вызывает `observe(...)`, затем
передаёт `PartnerState` в `loop.policy_context(partner=...)`. `partner_model=None`
→ контур S4 идентичен. Сигнатуры живут in-memory в сессии (персистентность в
общий store — отложено).

**Vigilance (S5):** `VigilanceGate.observe` маркирует утверждение как
гипотезу при конфликте с накопленным; ответ **не** блокируется. Команда
`/name X` привязывает объявленное имя к сигнатуре; имя попадает в
system-промпт (`IntentFrame.partner_name`). `JointAgency` (режим
`cooperative`) отслеживает совместную цель и возвращает напоминание при уходе.
Социальные метрики уходят в телеметрию через `loop.record_social(...)`.
Команды `/clear` (очистить историю), `/quit`. Реплики хоста **не** пишутся в
память (S3). `input_fn`/`output_fn` инъектируются для тестов.

При `show_status=True` (CLI `--status`) перед каждой репликой печатается
строка состояния из `format_status`:

```
[F=18.06 val=-16.83 stress=1.36 γ=9.95 задача=tone recall=1 дрейф=нет]
```

Поля: F, valence, stress, γ, активная задача, найден ли прецедент, дрейф.
Нужна для HITL-валидации ворот S3 («тон следует аффекту»). Полноценная
панель (графики, pause/step/inject) — S4 (control channel).

## Инварианты

1. Речь не роняет контур: ошибки LLM/recall → `None`/лог.
2. Event-triggered: LLM не вызывается на каждый тик.
3. `intent.py` — чистый Core (детерминирован, без I/O).
4. Правки запоминаются: новое сообщение → эпизод → recall.
5. `speech.enabled=False` → контур S1/S2 без изменений.
