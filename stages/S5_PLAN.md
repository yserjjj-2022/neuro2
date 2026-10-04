# PLAN.md — S5: Социальность

Реализация `stages/S5_SPEC.md`. Порядок — два прохода: **проход 1** даёт
минимальный социальный контур (сигнатура, узнавание, тайминг, ToM → policy),
**проход 2** расширяет (Vigilance Gate, имена, Joint Agency, телеметрия).
Каждый шаг — сначала Core (чистые функции + тесты), затем Shell (wiring/chat).

Модульные SPEC (`src/tm/`, `src/host/`, `src/speech/`, `src/config/`,
`src/telemetry/`) обновляются по ходу. Новый модуль `src/tm/` требует
`SPEC.md` + `PLAN.md` + `README.md` до кода (CONSTITUTION §5.1).

## Файлы

### Новые (проход 1)

1. `src/tm/` — `models.py` (`PartnerState`, `PartnerSignature`),
   `compute.py` (`match_partner`, `update_signature`, `update_trust` — Core),
   `partner.py` (`PartnerModel` — Shell), `__init__.py`, `SPEC.md`, `PLAN.md`,
   `README.md`
2. `src/host/sources.py` — `PauseProvider` (провайдер тайминга)
3. `src/tests/test_tm_compute.py`
4. `src/tests/test_tm_partner.py`
5. `src/tests/test_host_pause_source.py`

### Новые (проход 2)

6. `src/tm/vigilance.py` — `Claim`, `VigilanceGate` (Shell) + `detect_conflict`
   (Core, в `compute.py`)
7. `src/tm/joint.py` — Joint Agency над `TaskAttractor` (Shell, минимально)
8. `src/tests/test_tm_vigilance.py`
9. `src/tests/test_tm_joint.py`
10. `src/tests/test_integration_social_loop.py`

### Изменяемые

1. `src/core/policy/models.py` — `PolicyContext.partner: PartnerState | None`
2. `src/core/policy/compute.py` — учёт `partner.uncertainty`/`trust` в
   оценке `IDENTIFY_PARTNER`/`RESPOND`
3. `src/config/params.py` — `SocialConfig`; поле в `HostConfig`; валидация
4. `src/config/__init__.py`, `src/config/SPEC.md` — re-export/доки
5. `src/telemetry/models.py` + `logger.py` — `partner_trust`,
   `partner_uncertainty`, `partner_name`, `pause_s`, `claim_conflict`
6. `src/telemetry/SPEC.md`
7. `src/speech/chat.py` — владение `PartnerModel`; обновление на реплике;
   передача `PartnerState` в policy
8. `src/host/loop.py` — `policy_context(partner=...)`; проброс `PauseProvider`
9. `src/host/SPEC.md`, `src/speech/SPEC.md` — порядок и связки
10. `src/__main__.py` — `--no-social`
11. `src/tests/test_config.py`, `test_telemetry_*`, `test_speech_policy.py`,
    `test_speech_chat.py` — адаптация + новые случаи
12. `BUILD_ROADMAP.md`, `VALIDATION.md`, `SPECS.md`, `BACKLOG.md` — синк

## Зависимости

- **Внешние:** только stdlib (`dataclasses`, `collections.abc`) и `numpy`
  (уже есть). Новых зависимостей нет.
- **Внутренние:** `tm` ← `memory` (Embedder/Episode/cosine), `mcp`
  (`SignalSource` для `PauseProvider`); `policy` ← `tm` (только тип
  `PartnerState`, без цикла — `PolicyContext` в core не импортирует `tm`;
  поле объявляется через `TYPE_CHECKING`/локальный протокол);
  `speech.chat` ← `tm`; `config` ← `tm`.

## Проход 1 — минимальный социальный контур (P0)

### Шаг 1. tm Core: сигнатура и матчинг

1. `PartnerSignature` (frozen): `centroid`, `weight`, `mean_pause_s`,
   `mean_valence`, `mean_stress`, `name`, `aliases`.
2. `match_partner(embedding, signatures, *, threshold) -> (index|None, sim)` —
   косинус (`memory.similarity.cosine_similarity`), порог, детерминизм.
3. `update_signature(signature|None, embedding, *, pause_s, valence, stress,
   learning_rate) -> PartnerSignature` — скользящее среднее; создание при None;
   чистая (не мутирует вход).
4. `update_trust(state, *, conflict, ambiguity, trust_gain, trust_decay)`
   — доверие растёт/утекает; остаётся в [0, 1].
5. Тесты: матчинг выше/ниже порога, пустой список → (None, 0.0), детерминизм,
   скользящее среднее сходится, границы [0, 1], чистота.

### Шаг 2. tm Shell: PartnerModel

1. `PartnerModel(embedder, *, match_threshold, learning_rate, ...)` —
   владеет `list[PartnerSignature]`.
2. `observe(text, *, pause_s, valence, stress) -> PartnerState`:
   embed → match → update/create signature → собрать `PartnerState`
   (`uncertainty = 1 - sim`, `trust` из `update_trust`).
3. `set_name(name, aliases=())` — привязка имени к текущей сигнатуре
   (объявление партнёра, ADR-0008 §4).
4. Ошибки embedder → лог + безопасный дефолт (`PartnerState` с
   `uncertainty=1.0`); не роняет чат.
5. Тесты: первая реплика создаёт сигнатуру, вторая близкая — матчится,
   далёкая — новая сигнатура (гистерезис), имя привязывается, сбой embedder.

### Шаг 3. PauseProvider

1. `PauseProvider(clock, tau_s)` — нормированная пауза с прошлой реплики.
2. `read(tick, now)` → `SignalSource(category=INTEROCEPTIVE, data=[pause_norm])`;
   `pause_norm = 1 - exp(-pause/tau)` ∈ [0, 1]; первая реплика → 0.
3. Регистрация в `default_providers` (опционально, через флаг/аргумент) —
   чтобы `bus_dim` учитывался колонками.
4. Тесты: нормировка, монотонность, детерминизм при фиксированных часах,
   dim/категория, инъекция часов.

### Шаг 4. PolicyContext + policy

1. `PolicyContext.partner: PartnerState | None = None` (расширяемое поле,
   S4-совместимость). Тип без цикла: `tm` — отдельный слой, `core.policy`
   не импортирует его напрямую (протокол/`TYPE_CHECKING`).
2. `_evaluate_identify`: при `partner.uncertainty >= identify_threshold` →
   эпистемическая ценность > 0 (мягкий интент, ADR-0008 §5); иначе 0.
3. `_evaluate_respond`: `partner.trust` масштабирует прагматическую ценность
   (адаптация под персону).
4. Тесты: uncertainty высокая → `IDENTIFY_PARTNER` выигрывает; `partner=None`
   → прежнее поведение S4; детерминизм; трасса объяснима.

### Шаг 5. Связка в ChatSession

1. `ChatSession` владеет `PartnerModel` (инъекция; None → ToM выключена).
2. На реплике: вычислить `pause_s` (wall-clock), `observe(...)` → `PartnerState`.
3. Передать `partner` в `loop.policy_context(...)`; при `identify_partner`
   goal доходит до IntentFrame.
4. Обратная совместимость: `partner_model=None` → контур S4 идентичен.
5. Тесты: сигнатура копится между ходами, узнавание во втором диалоге
   (после reopen store), `identify_partner` при высокой uncertainty.

### Шаг 6. Валидация прохода 1

1. Интеграционный тест: два диалога → одна сигнатура; имя сохраняется.
2. `social.enabled=False` / `partner_model=None` → S4 идентичен.
3. Обновить `behavioral_fingerprint` осознанно (новые поля).
4. Синк `VALIDATION.md` §4, `BUILD_ROADMAP.md` §1, `SPECS.md`, `BACKLOG.md`.

## Проход 2 — расширение (P1)

### Шаг 7. Vigilance Gate

1. `Claim` (frozen): `content`, `confidence`, `conflict`, `confirmed`.
2. `detect_conflict(claim_embedding, memory_embeddings, *, threshold) -> float`
   — максимум `1 - cos` по накопленному (чистая).
3. `VigilanceGate(store, embedder, threshold)` — Shell: новое утверждение →
   `Claim` с confidence по конфликту; подтверждение практикой повышает
   confidence.
4. Инвариант: не блокирует ответ; поведение — по сырому сигналу.
5. Тесты: конфликтное утверждение → низкая confidence, согласное → высокая;
   подтверждение; отсутствие блокировки; детерминизм.

### Шаг 8. Имена и алиасы

1. `PartnerModel.set_name(name, aliases)`; `resolve_name()` для вокатива.
2. Имя — первичный ключ поверх сигнатуры (устойчиво сквозь дрейф).
3. Тесты: объявление имени, алиасы, вокатив в промпте (register/style).

### Шаг 9. Joint Agency (минимально)

1. `JointGoal` над `TaskAttractor`: общая цель с TTL/приоритетом.
2. При уходе партнёра от общей цели — реакция (напоминание/подстраховка),
   не исполнение приказа; активна в режиме `cooperative`.
3. Тесты: совместная цель держится; уход → реакция; не приказ.

### Шаг 10. Телеметрия

1. +5 полей: `partner_trust`, `partner_uncertainty`, `partner_name`,
   `pause_s`, `claim_conflict` (keyword-only с дефолтами).
2. Тесты: сериализация (28 полей), дефолты, обратная совместимость `log()`.

### Шаг 11. CLI + конфиг

1. `--no-social` (S4-совместимость); `--mode cooperative` для Joint Agency.
2. `SocialConfig` в `HostConfig`; валидация порогов.
3. Тесты: `--no-social` отключает ToM; валидация конфига.

## План тестов

| Тест | Покрытие | Инвариант |
|---|---|---|
| `test_tm_match_partner_*` | матчинг Core | порог, детерминизм |
| `test_tm_update_signature` | сигнатура Core | сходимость, чистота |
| `test_tm_update_trust` | trust Core | границы [0, 1] |
| `test_partner_model_observe` | tm Shell | сигнатура копится |
| `test_partner_model_name` | имена | привязка к сигнатуре |
| `test_pause_provider_*` | тайминг | нормировка, инъекция часов |
| `test_policy_identify_uncertain` | policy | мягкий интент |
| `test_policy_partner_none_s4_compat` | совместимость | S4 идентичен |
| `test_chat_signature_accumulates` | связка | непрерывность |
| `test_chat_recognition_reopen` | узнавание | та же сигнатура |
| `test_vigilance_conflict_claim` (п.2) | C9 | гипотеза |
| `test_joint_agency_*` (п.2) | совместная цель | не приказ |
| `test_telemetry_social_fields` (п.2) | 28 полей | сериализация |
| `test_social_disabled_s4_compat` (п.2) | совместимость | S4 идентичен |

## Заметки

- **Порядок:** Core (чистые функции) → Shell (PartnerModel) → wiring (chat).
  Матчинг/обновление — детерминированные чистые функции (тестируемы без БД).
- **Речь ≠ тик:** ToM обновляется в `ChatSession` (вне тика), как policy;
  медленный эмбеддинг не блокирует аффективный контур.
- **Память:** сигнатуры — домен общей памяти; переиспользовать `Episode`/
  `Embedder`/`cosine_similarity`, не плодить хранилищ (ADR-0008 §3).
- **Без цикла импортов:** `core.policy` не импортирует `tm`; `PartnerState`
  в `PolicyContext` — через `TYPE_CHECKING` или Protocol. `tm` зависит от
  `memory`/`mcp`, не наоборот.
- **Расширяемость:** `PolicyContext.partner` — дефолт `None`; S6 добавляет
  метакогницию так же (решение A, S4_SPEC).
- **Обратная совместимость:** `social.enabled=False` / `partner_model=None`
  → контур S4 идентичен; `--no-social` в CLI.
- **Гистерезис:** порог матчинга + требование устойчивости во времени
  защищают от ложных персон (ADR-0008).
- **Reference:** `MemoryRouter` (Shell с DI), `SpeechController` (событийный
  Shell), `Homeostat` (Core+Shell FC/IS), `Embedder` (env+auto+fake).
