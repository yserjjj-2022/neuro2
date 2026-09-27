# PLAN.md — src/core/homeostasis

Реализация `src/core/homeostasis/SPEC.md` (S4, проход 1).

## Файлы

- `models.py` — `Setpoint`, `HomeostaticSignal`, `HomeostasisState` (frozen)
- `compute.py` — `setpoint_deviation` (Core, чистая)
- `manager.py` — `Homeostat` (Shell)
- `__init__.py` — re-export
- `src/tests/test_homeostasis_compute.py`, `test_homeostasis_manager.py`

## Порядок

1. `Setpoint` + валидация.
2. `setpoint_deviation` (clip, `ValueError`).
3. `HomeostaticSignal` / `HomeostasisState`.
4. `Homeostat.evaluate`.
5. Тесты (границы, порядок, агрегаты, детерминизм).
6. Re-export в `src/core/__init__.py`.

## Заметки

- `critical` = порог reflex (0.9) — согласованность отклонения и критичности.
- `weight` заложен под корневые приоры (S5/S6), в S4 не влияет на агрегат.
