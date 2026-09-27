# PLAN.md — src/core/policy

Реализация `src/core/policy/SPEC.md` (S4, проход 1).

## Файлы

- `models.py` — `Action`, `Preferences`, `PolicyContext`, `PolicyCandidate`,
  `PolicyTrace` (frozen)
- `compute.py` — `evaluate_candidates`, `select_action` (Core, чистые)
- `__init__.py` — re-export
- `src/tests/test_policy_compute.py`

## Порядок

1. `Action` (Enum), `Preferences` (+валидация), `PolicyContext`.
2. `PolicyCandidate`, `PolicyTrace`.
3. `evaluate_candidates` — правила по действиям.
4. `select_action` — детерминированный argmax + причина.
5. Тесты: goal-directed, детерминизм, трасса, валидация.
6. Re-export в `src/core/__init__.py`.

## Заметки

- Порядок `_ACTION_ORDER` задаёт тай-брейк (детерминизм).
- `ALERT_DEVIATION=0.7` — порог тревоги (калибровка — Open Question S4).
- S6 добавляет метакогницию полем `PolicyContext` без правки функций.
