# PLAN.md — src/core/actuation

Реализация `src/core/actuation/SPEC.md` (S8, этап 1).

## Файлы

- `models.py` — `OptionSource`, `Option`, `OptionWindow`, `OptionContext`,
  `ActuationPreferences`, `OptionCandidate`, `OptionTrace` (frozen, с валидацией)
- `compute.py` — `build_options`, `score_option`, `select_option` (Core, чистые)
- `__init__.py` — re-export
- `src/tests/test_actuation.py`

## Порядок

1. `OptionSource` (Enum), `Option` (+валидация), `OptionWindow` (+ `find`/`ids`/`tools`).
2. `OptionContext`, `ActuationPreferences` (+валидация весов/порогов).
3. `OptionCandidate`, `OptionTrace`.
4. `build_options` — из `Affordance` → `Option(source=TOOL)`, порядок стабилен.
5. `score_option` — тотален: TOOL → pragmatic=0, epistemic=uncertainty·relevance (если relevance ≥ floor).
6. `select_option` — детерминированный argmax + тай-брейк (индекс в окне), полная трасса.
7. Тесты: открытость окна, тотальность скорера, детерминизм/тай-брейк, необратимая занижена, валидация.
8. Re-export в `src/core/__init__.py`.

## Заметки

- `build_options` принимает `Affordance` из `src.mcp.probe` — **не** копирует модель.
- `score_option` — чистая функция: нет побочных эффектов, нет доступа к runtime-состоянию.
- `irreversible_penalty=0.5` — стартовая величина (SPEK §Open Questions).
- `relevance` как обогащение: если `None` → дефолт `uncertainty`; если задан — `uncertainty·relevance`.
