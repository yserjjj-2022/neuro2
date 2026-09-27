# src/core/policy — выбор действия (S4)

Policy / action selection (манифест §3.И): оценка кандидатов-действий по
прагматической и эпистемической ценности + **обязательная причинная трасса**
(explainability).

- `models.py` — `Action`, `Preferences`, `PolicyContext`, `MacroContext`,
  `PolicyCandidate`, `PolicyTrace`.
- `compute.py` — `evaluate_candidates`, `select_action` (чистые).

На S4 — только речевое поведение (respond/silent/initiative/identify_partner).
Потребитель: `speech` (`IntentFrame.goal`). Подробности — `SPEC.md`.
