# PLAN.md — S4: Воля

Реализация `stages/S4_SPEC.md`. Порядок — два прохода: **проход 1**
закрывает ворота S4 минимальным контуром, **проход 2** расширяет
(γ-барьер, control, gate, макро-контекст). Каждый шаг — сначала Core
(чистые функции + тесты), затем Shell (wiring/loop).

Модульные SPEC (`src/core/homeostasis/`, `src/core/policy/`,
`src/host/SPEC.md`, `src/config/SPEC.md`, `src/telemetry/SPEC.md`)
обновляются по ходу.

## Файлы

### Новые (проход 1)

1. `src/core/homeostasis/` — `models.py` (`Setpoint`, `HomeostaticSignal`,
   `HomeostasisState`), `compute.py` (`setpoint_deviation` — Core),
   `manager.py` (`Homeostat` — Shell), `__init__.py`, `SPEC.md`, `PLAN.md`
2. `src/core/policy/` — `models.py` (`Action`, `Preferences`, `PolicyContext`,
   `PolicyCandidate`, `PolicyTrace`), `compute.py` (`evaluate_candidates`,
   `select_action` — Core), `__init__.py`, `SPEC.md`, `PLAN.md`
3. `src/host/throttle.py` — `ThrottlePlan`, `plan_throttle` (Core)
4. `src/tests/test_homeostasis_compute.py`
5. `src/tests/test_homeostasis_manager.py`
6. `src/tests/test_policy_compute.py`
7. `src/tests/test_host_throttle.py`
8. `src/tests/test_integration_policy_loop.py`

### Новые (проход 2)

9. `src/core/cmc/attention.py` — пред-колоночный барьер `a(γ)` (Core)
10. `src/host/gate.py` — `CapabilityTier`, `ActionRequest`, `GateDecision`,
    `CapabilityGate` (Shell, заготовка)
11. `src/host/control.py` — `ControlChannel` (Shell)
12. `src/host/macro.py` — `MacroContext` (Core) *(либо в policy/models.py)*
13. `src/tests/test_cmc_attention.py`
14. `src/tests/test_host_gate.py`
15. `src/tests/test_host_control.py`

### Изменяемые

1. `src/core/__init__.py` — re-export homeostasis/policy
2. `src/config/params.py` — `HomeostasisConfig`, `PolicyConfig`; поля в
   `HostConfig`; валидация
3. `src/config/__init__.py`, `src/config/SPEC.md` — re-export/доки
4. `src/telemetry/models.py` + `logger.py` — `policy_action`, `policy_reason`,
   `throttle`, `homeostasis` (23 поля)
5. `src/telemetry/SPEC.md`
6. `src/host/loop.py` — гомеостаз → throttle → (рефлекс) → policy-контекст;
   логирование
7. `src/host/SPEC.md` — порядок тика
8. `src/speech/controller.py` — приём `goal`/`PolicyTrace`
9. `src/speech/chat.py` — вызов policy вместо/поверх `should_speak`;
   LLM-гейт при throttle
10. `src/speech/SPEC.md`
11. `src/__main__.py` — `--policy`/`--no-policy`, `--mode`; (проход 2) control
12. `src/tests/test_config.py`, `test_telemetry_*`, `test_speech_*` —
    адаптация + новые случаи
13. `BUILD_ROADMAP.md`, `VALIDATION.md`, `SPECS.md`, `BACKLOG.md` — синк

## Зависимости

- **Внешние:** только stdlib (`enum`, `dataclasses`, `collections.abc`) и
  `numpy` (уже есть). Новых зависимостей нет.
- **Внутренние:** `homeostasis` ← `mcp` (`SignalSource`); `policy` ←
  `homeostasis`; `host.loop` ← homeostasis/policy/throttle; `speech` ←
  policy; `config` ← homeostasis/policy.

## Проход 1 — ворота (P0)

### Шаг 1. Гомеостаз (Core + Shell)

1. `Setpoint` (frozen) + валидация `0 <= comfort < critical <= 1`,
   `weight > 0`.
2. `setpoint_deviation(severity, comfort, critical) -> float` — чистая,
   `clip` в [0, 1], `ValueError` при `critical <= comfort`.
3. `HomeostaticSignal`, `HomeostasisState` (frozen снимки).
4. `Homeostat(setpoints, reflex_threshold).evaluate(signals)` — Shell:
   читает severity из `SignalSource`, порядок = порядок сетпоинтов.
5. Тесты: границы deviation (comfort→0, critical→1, между), валидация
   сетпоинта, отсутствие канала в сигналах, `is_critical` при ≥ 0.9,
   чистота, детерминизм.

### Шаг 2. Policy (Core)

1. `Action` (Enum): respond/silent/initiative/identify_partner.
2. `Preferences`, `PolicyContext` (расширяемый), `PolicyCandidate`,
   `PolicyTrace` (frozen).
3. `evaluate_candidates(context, preferences) -> tuple[PolicyCandidate, ...]`
   — чистая; правила из SPEC §2.
4. `select_action(context, preferences) -> PolicyTrace` — детерминированный
   argmax, тай-брейк по порядку `Action`; причина выводится из оценок.
5. Тесты: goal-directed (смена `respond_to_messages`/`homeostatic_alert` →
   другое действие), детерминизм, тай-брейк, `SILENT` при сбое (Shell-уровень),
   трасса содержит всех кандидатов, explainability (reason непустой и
   согласован с победителем).

### Шаг 3. Throttle + reflex-путь в loop

1. `ThrottlePlan` (frozen), `plan_throttle(homeostasis, ...)` — чистая.
2. `HostLoop`: `homeostat` в конструкторе; в `step_once` —
   `homeostat.evaluate(bus.last_signals)` → `plan_throttle` → применение
   `dt_eff`/`k_eff` в том же тике.
3. LLM-гейт: `HostLoop.last_throttle` доступен `ChatSession`.
4. Тесты: критический сигнал → `throttle.active` на том же/следующем тике;
   `active=False` в норме; C6 (перегруз → throttle); обратная совместимость
   (без сетпоинтов throttle неактивен).

### Шаг 4. Policy-контекст в loop

1. `HostLoop` собирает `PolicyContext` (F/valence/stress/task/гомеостаз/
   has_new_message/mode) после `pipeline.tick`.
2. Телеметрия: `policy_action`, `policy_reason`, `throttle`, `homeostasis`.
3. Тесты: поля сериализуются (23); `policy_action` пуст при `enabled=False`.

### Шаг 5. Связка policy → речь

1. `SpeechController.respond(..., goal=None)` → `build_intent_frame(goal=...)`.
2. `ChatSession`: `select_action` → `goal`; при `throttle.llm_gate` инициатива
   запрещена (ответ на сообщение сохраняется).
3. `policy.enabled=False` → прежний `should_speak`.
4. Тесты: goal доходит до frame; silent → `[хост промолчал]`; LLM-гейт
   блокирует инициативу, но не ответ; S3-совместимость.

### Шаг 6. Валидация ворот

1. Goal-directed test как интеграционный тест (смена `Preferences`).
2. Рефлекс ≤ 1 тик (телеметрия).
3. Explainability (трасса в телеметрии).
4. C6 throttle.
5. Обновить `behavioral_fingerprint` (новые поля/поведение) осознанно.
6. Синхронизировать `VALIDATION.md` §4 (S4 — чекбоксы), `BUILD_ROADMAP.md`
   (§1 таблица звеньев: решение/действие/гомеостаз/самоконтроль),
   `SPECS.md`, `BACKLOG.md` (метакогниция → S6).

## Проход 2 — расширение (P1)

### Шаг 7. γ пред-колоночно

1. `a(γ)` в `src/core/cmc/attention.py` (Core, формула — Open Question).
2. `CMCPipeline.tick(..., attention=None)`: `u_eff = u · a(γ)` при флаге.
3. `attention_gate=False` → `u_eff = u` (S1–S3 идентичны).
4. Тесты: монотонность, границы, отключённый gate = тождество.

### Шаг 8. Capability gate (заготовка)

1. `CapabilityTier`, `ActionRequest`, `GateDecision`, `CapabilityGate`.
2. `request()`: tier ≤ max_tier → allow; иначе fail-safe deny.
3. Audit: `ActionRequest.reason` логируется.
4. Тесты: allow T1, deny T2 при `max_tier=T1`, deny при таймауте HITL.

### Шаг 9. Control channel

1. `ControlChannel(loop)`: `status`, `pause`, `resume`, `step`.
2. `status` переиспользует `format_status`.
3. CLI: (опц.) `--control` интерактивный режим.
4. Тесты: pause блокирует `run`, step делает ровно n, status непустой.

### Шаг 10. Макро-контекст

1. `MacroContext(task, mode)`; `PolicyContext.mode` из конфига.
2. `--mode {game,cooperative,free}` в CLI.
3. Тесты: mode доходит в контекст; валидация значения.

## План тестов

| Тест | Покрытие | Инвариант |
|---|---|---|
| `test_setpoint_deviation_*` | гомеостаз Core | границы [0,1] |
| `test_homeostat_evaluate` | гомеостаз Shell | порядок, отсутствие канала |
| `test_policy_goal_directed` | policy Core | goal-directed |
| `test_policy_deterministic` | policy Core | детерминизм/тай-брейк |
| `test_policy_trace_explainable` | policy Core | explainability |
| `test_plan_throttle_*` | throttle Core | активен при critical |
| `test_loop_reflex_one_tick` | loop | reflex ≤ 1 тик |
| `test_loop_throttle_c6` | loop | C6 |
| `test_telemetry_policy_fields` | 23 поля | сериализация |
| `test_speech_goal_from_policy` | связка | goal в frame |
| `test_speech_llm_gate` | связка | инициатива под throttle |
| `test_policy_disabled_s3_compat` | совместимость | S3 идентичен |
| `test_cmc_attention_*` (п.2) | γ-барьер | монотонность/тождество |
| `test_gate_fail_safe_deny` (п.2) | gate | deny |
| `test_control_*` (п.2) | control | pause/step/status |

## Заметки

- **Порядок в тике:** гомеостаз → throttle (рефлекс) → pipeline. Throttle
  влияет на `dt`/`k` текущего тика; policy — на речь (вне тика).
- **Речь ≠ тик:** `select_action` вызывается в `ChatSession`, не в
  `step_once` — медленный LLM не блокирует аффективный контур.
- **Расширяемый контекст:** `PolicyContext` — frozen dataclass; S6
  добавляет метакогницию полем с дефолтом.
- **Explainability:** `PolicyTrace.reason` и `ThrottlePlan.reason` выводятся
  из расчёта; обязательны, не best-effort (манифест §3.И).
- **Fail-safe deny:** gate отказывает по умолчанию (неизвестный tier/таймаут).
- **Обратная совместимость:** `policy.enabled=False` → `should_speak` (S3);
  `attention_gate=False` → вход колонок не изменён.
- **Mock-оговорка:** throttle на S4 наблюдаем (контракт + аудит) и реально
  влияет на дорогой путь (LLM-гейт); физический эффект на латентность —
  по мере тяжёлых операций (S5+).
- **Reference:** `DriftDetector`/`PrecisionEstimator` (energy) — Shell с
  состоянием; `SpeechController` — событийный Shell; `TaskAttractor` —
  FC/IS Core+Shell; `Embedder` — env+auto+fake (для CLI-ручек, если нужны).
