# src/core/homeostasis — гомеостаз (S4)

Интероцептивные сетепоинты: сравнение `severity` сигналов шины с целевым
диапазоном и нормированное отклонение. Манифест §3.В/§3.З.

- `models.py` — `Setpoint`, `HomeostaticSignal`, `HomeostasisState`.
- `compute.py` — `setpoint_deviation` (чистая).
- `manager.py` — `Homeostat` (Shell).

Потребители: `core/policy` (речевая тревога через `max_deviation`),
`host/throttle` (reflex через `is_critical`). Подробности — `SPEC.md`.
