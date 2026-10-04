# PLAN.md — S7: HITL-диагностика

Реализация `stages/S7_SPEC.md`. Решения — ADR-0010. Порядок: сначала
формальный harness (числа/ручки) и пресеты (база для воспроизводимости), затем
диалоговая сессия (категории/ветвление), затем самоотчёт сброса и протокол.
Каждый шаг — Core (чистые + тесты) → Shell (wiring) → CLI.

Новые модули требуют `SPEC.md` + `README.md` до кода (CONSTITUTION §5.1).

## Файлы

### Новые

1. `src/host/fingerprint.py` — `BehavioralFingerprint`,
   `behavioral_fingerprint`, `fingerprint_distance` (Core)
2. `src/host/presets.py` (или `src/config/presets.py`) — фабрики пресетов +
   `load_preset` (гибрид Python+TOML)
3. `configs/dialogue.toml`, `configs/stress.toml` — примеры override
4. `src/host/sensitivity.py` — `SensitivityCase`, `build_sensitivity_matrix`,
   `check_direction` (Core) + `SensitivityRunner` (Shell)
5. `src/host/diagnostic.py` — `Verdict`, `Probe`, `ProbeResult`,
   `DiagnosticSnapshot`, `next_probe` (Core) + `DiagnosticSession` (Shell)
6. `src/host/probes.py` — дерево проб (данные) для S3–S6
7. `src/speech/intent.py` (доп.) — `reset_self_report` (Core)
8. `src/tests/test_sensitivity.py`
9. `src/tests/test_presets.py`
10. `src/tests/test_diagnostic.py`
11. `src/tests/test_reset_self_report.py`
12. `src/host/fingerprint/SPEC.md` + `README.md` (или в `src/host/SPEC.md`)
13. `stages/S7_HITL_PROTOCOL.md` — операторский протокол

### Изменяемые

1. `src/tests/test_behavioral_regress.py` — импорт fingerprint из Core
2. `src/host/control.py` — `snapshot()` (полный снимок для диагностики)
3. `src/__main__.py` — `--preset`, `--preset-file`, `--sensitivity`,
   `--diagnose`, `--diagnose-log`, `--probes`
4. `src/host/SPEC.md`, `src/config/SPEC.md`, `src/speech/SPEC.md` — новые API
5. `src/tests/test_config.py` — пресеты
6. `BUILD_ROADMAP.md`, `VALIDATION.md`, `SPECS.md`, `BACKLOG.md`, `README.md` — синк

## Зависимости

- **Внешние:** stdlib + numpy. TOML — `tomllib` (stdlib). **Новой зависимости
  нет.**
- **Внутренние:** `fingerprint` ← telemetry; `presets` ← config; `sensitivity`
  ← fingerprint + loop; `diagnostic` ← loop + selfcontrol + gate + presets;
  `probes` — данные; `reset_self_report` ← selfcontrol models.

## Шаг 1. Fingerprint Core (S7-A)

1. Перенести `behavioral_fingerprint` из `src/tests/test_behavioral_regress.py`
   в `src/host/fingerprint.py`; добавить `BehavioralFingerprint` (frozen) и
   `fingerprint_distance`.
2. Re-export в тестовом хелпере; существующие тесты не меняют поведение.
3. Тесты: отпечаток детерминирован; пустой вход → ValueError; расстояние
   симметрично, 0 при равенстве, метрики в границах.

## Шаг 2. Sensitivity harness (S7-A)

1. Core: `SensitivityCase`, `build_sensitivity_matrix` (хардкод ручек/диапазонов/
   направлений), `check_direction` (инвариант направления, `bounded`).
2. Shell: `SensitivityRunner.run_case`/`run_all` — прогон loop под пресетом,
   сбор отпечатков.
3. pytest `test_sensitivity.py`: направления выполняются; тот же seed → тот же
   отпечаток; метрики в границах.
4. CLI `--sensitivity`: печать матрицы (ручка → метрики → вердикт).

## Шаг 3. Пресеты (S7-B)

1. `presets.py`: `baseline`, `stress`, `dialogue`, `autonomy`, `long_horizon`,
   `cooperative` — детерминированные `HostConfig`-фабрики.
2. `load_preset(name, *, override=None)`: база + TOML-override; fail-fast при
   неизвестном ключе/недопустимом значении.
3. `configs/dialogue.toml`, `configs/stress.toml` — примеры.
4. CLI: `--preset NAME [--preset-file PATH]`.
5. Тесты: база детерминирована; override перезаписывает поля; неизвестный ключ
   → ValueError; неизвестный пресет → ValueError.

## Шаг 4. Диагностический движок (S7-C)

1. Core: `Verdict`, `Probe`, `ProbeResult`, `DiagnosticSnapshot`, `next_probe`.
2. Shell: `DiagnosticSession` (проба → снимок → вопрос → вердикт → ветвление →
   журнал); инъекция `input_fn`/`output_fn`/`clock`.
3. `control.py`: `snapshot()` — полный снимок (F/val/stress/γ/задача/партнёр/
   метакогниция/сброс) для `DiagnosticSnapshot`.
4. Журнал JSONL (`--diagnose-log`): `{probe_id, verdict, comment, snapshot,
   next_probe}`.
5. CLI: `--diagnose [--preset NAME] [--probes PATH]`.
6. Тесты: ветвление по вердикту; журнал пишется; вердикт несёт snapshot;
   детерминизм Core; скриптованный ввод (без реального человека).

## Шаг 5. Дерево проб (S7-C)

1. `probes.py`: начальный набор проб S3–S6 с ветвлениями и fallback.
2. Покрытие: S3 тон/уместность/память; S4 рефлекс/escape hatch/explainability;
   S5 узнавание/гипотеза/имя; S6 наблюдаемые/сброс/консолидация/самоотчёт.
3. Тесты: все id уникальны; ветвления ссылаются на существующие пробы;
   fallback задан; стартовая проба существует.

## Шаг 6. Самоотчёт сброса (S7-D)

1. Core: `reset_self_report(*, plan, task)` — строка при `triggered=True`,
   `None` иначе.
2. Речевой интент `report_reset` в `ChatSession`/`SpeechController`.
3. Исполнение сброса через `CapabilityGate`: необратимо → `hitl_token`;
   таймаут → deny (fail-safe).
4. Тесты: `triggered=False` → None; `HARD` → обязательный отчёт без автосброса;
   deny без токена; allow с токеном.

## Шаг 7. Операторский протокол (S7-E)

1. `stages/S7_HITL_PROTOCOL.md`: подготовка, порядок прогонов, дерево проб,
   чек-лист по стадиям, нормировка, границы, мост в машинный тест.
2. Синк `VALIDATION.md` §6 (ссылка на протокол), `SPECS.md` (реестр S7),
   `BUILD_ROADMAP.md` §3 (стадия S7), `BACKLOG.md` `[HITL][docs]`,
   `README.md`.
3. Ручной прогон протокола как приёмка ворот S7.

## План тестов

| Тест | Покрытие | Инвариант |
|---|---|---|
| `test_fingerprint_*` | Core | детерминизм, границы, пустой вход |
| `test_sensitivity_direction_*` | Core | инварианты направления |
| `test_sensitivity_runner_*` | Shell | seed → отпечаток, bounded |
| `test_preset_*` | Core+Shell | детерминизм базы, override, fail-fast |
| `test_probe_branching_*` | Core | ветвление по вердикту |
| `test_probe_tree_valid` | данные | id уникальны, ссылки валидны |
| `test_diagnostic_session_*` | Shell | журнал, snapshot, скриптованный ввод |
| `test_reset_self_report_*` | Core+Shell | триггер/None, HARD, deny/allow |
| `test_s7_disabled_compat` | совместимость | без флагов S7 контур S6 идентичен |

## Заметки

- **Порядок:** fingerprint → harness → пресеты → движок → пробы → самоотчёт →
  протокол.
- **Без новых зависимостей:** TOML — stdlib.
- **Ручки:** крутятся только в harness; диалог заморожен пресетом (ADR-0010 §3).
- **Категории:** человек даёт `matches/partial/mismatch`; числа — snapshot.
- **Fail-safe:** исполнение сброса без HITL-токена → deny.
- **Reference:** `ChatSession` (Shell+инъекция), `ControlChannel` (snapshot),
  `CapabilityGate` (HITL), `SelfMonitor`/`ResetPlan` (S6),
  `behavioral_fingerprint` (тесты).
