# SPECS — реестр спецификаций проекта

Единая точка навигации: какой модуль, где его SPEC/PLAN/README и на какой
стадии он находится. Модульные спеки живут **рядом с кодом** (CONSTITUTION §5.1);
этот файл — только индекс.

Сквозные документы (не принадлежат модулю):
- [`CONSTITUTION.md`](CONSTITUTION.md) — правила проекта
- [`host_architecture_manifest.md`](host_architecture_manifest.md) — архитектура (что строим)
- [`BUILD_ROADMAP.md`](BUILD_ROADMAP.md) — порядок сборки S1–S6
- [`VALIDATION.md`](VALIDATION.md) — проверка и ворота
- [`BACKLOG.md`](BACKLOG.md) — задачи
- [`adr/`](adr/) — Architecture Decision Records (0001–0008)

---

## Реестр модулей

| Модуль | SPEC | PLAN | README | Статус | Стадия |
|---|---|---|---|---|---|
| `core/cmc` | [SPEC](src/core/cmc/SPEC.md) | [PLAN](src/core/cmc/PLAN.md) | [README](src/core/cmc/README.md) | ✅ реализовано | S1 |
| `core/energy` | [SPEC](src/core/energy/SPEC.md) | [PLAN](src/core/energy/PLAN.md) | [README](src/core/energy/README.md) | ✅ реализовано (S1) | S1 |
| `core/voting` | [SPEC](src/core/voting/SPEC.md) | [PLAN](src/core/voting/PLAN.md) | [README](src/core/voting/README.md) | ✅ реализовано | S1 |
| `core/attractors` | [SPEC](src/core/attractors/SPEC.md) | [PLAN](src/core/attractors/PLAN.md) | [README](src/core/attractors/README.md) | ✅ реализовано (опережение) | S4 |
| `core/homeostasis` | [SPEC](src/core/homeostasis/SPEC.md) | [PLAN](src/core/homeostasis/PLAN.md) | [README](src/core/homeostasis/README.md) | ✅ реализовано (S4) | S4 |
| `core/policy` | [SPEC](src/core/policy/SPEC.md) | [PLAN](src/core/policy/PLAN.md) | [README](src/core/policy/README.md) | ✅ реализовано (S4) | S4 |
| `memory` | [SPEC](src/memory/SPEC.md) | [PLAN](src/memory/PLAN.md) | [README](src/memory/README.md) | ✅ store/recall + эмбеддер/приор/роутер (S2) | S2 |
| `telemetry` | [SPEC](src/telemetry/SPEC.md) | [PLAN](src/telemetry/PLAN.md) | [README](src/telemetry/README.md) | ✅ реализовано (24 поля) | S1–S4 |
| `mcp` | [SPEC](src/mcp/SPEC.md) | [PLAN](src/mcp/PLAN.md) | [README](src/mcp/README.md) | 🟡 контракт; transport ✗ | S1/S2 |
| `host` | [SPEC](src/host/SPEC.md) | [PLAN](src/host/PLAN.md) | [README](src/host/README.md) | ✅ реализовано (S1/S2/S4) | S1/S2/S4 |
| `config` | [SPEC](src/config/SPEC.md) | [PLAN](src/config/PLAN.md) | [README](src/config/README.md) | ✅ реализовано (S1/S2/S3/S4) | S1–S4 |
| `speech` | [SPEC](src/speech/SPEC.md) | [PLAN](src/speech/PLAN.md) | [README](src/speech/README.md) | ✅ реализовано (S3/S4) | S3/S4 |
| `tm` | [SPEC](src/tm/SPEC.md) | [PLAN](src/tm/PLAN.md) | [README](src/tm/README.md) | ✅ реализовано (S5) | S5 |
| `tests` | — | — | [README](src/tests/README.md) | ✅ pytest (644) | все |

SPEC стадии — `stages/S5_SPEC.md`, `stages/S5_PLAN.md`.

**Легенда:** ✅ реализовано · 🟡 частично · ✗ не начато · ⛔ заблокировано.

---

## Порядок стадий (ADR-0005)

`S1` честные сигналы → `S2` непрерывность (память+эмбеддер) → `S3` голос ✅ →
`S4` воля ✅ (проход 1 — ворота; проход 2 — γ-барьер, gate, control,
макро-контекст; долги аудита закрыты) → `S5` социальность (ToM) ✅ (сигнатура/
узнавание/тайминг/ToM→policy, Vigilance, имена, Joint Agency) → `S6` автономия.

Подробно — [`BUILD_ROADMAP.md`](BUILD_ROADMAP.md); ворота — [`VALIDATION.md`](VALIDATION.md).

---

## Правило добавления

Новый модуль → создать `src/<path>/SPEC.md` + `PLAN.md` + `README.md`
до кода (CONSTITUTION §5.1), затем добавить строку в таблицу выше.
