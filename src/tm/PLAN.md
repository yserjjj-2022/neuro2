# PLAN.md — src/tm

Реализация `src/tm/SPEC.md` (S5, проход 1).

## Файлы

- `models.py` — `PartnerState`, `PartnerSignature` (frozen)
- `compute.py` — `match_partner`, `update_signature`, `update_trust` (Core)
- `partner.py` — `PartnerModel` (Shell)
- `__init__.py` — re-export
- `src/tests/test_tm_compute.py`, `src/tests/test_tm_partner.py`

## Порядок

1. `PartnerState` (+валидация [0, 1]), `PartnerSignature`.
2. `match_partner` — косинус, порог, пустой список.
3. `update_signature` — скользящее среднее, создание при None, чистота.
4. `update_trust` — доверие с утечкой/приростом, clamp.
5. `PartnerModel` — embed → match → update/create; `set_name`.
6. Тесты: матчинг, сходимость, чистота, границы, сбой эмбеддера, имя.
7. Re-export в `src/tm/__init__.py`.

## Заметки

- Пороги/learning_rate — из `SocialConfig` (S5); в модуле — параметры с
  дефолтами (валидируются).
- `Vector` берётся из `src.memory.serialize` (общий тип).
- `cosine_similarity` переиспользуется из `memory` — не дублировать.
- `observe` не роняет чат: `EmbedderError` → лог + дефолт.
- Проход 2 (выполнен): `vigilance.py` (`Claim`, `VigilanceGate`,
  `detect_conflict`), `joint.py` (`JointGoal`, `JointAgency`), вокатив имени
  (`IntentFrame.partner_name` → system-промпт).
