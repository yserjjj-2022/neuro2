# SPEC.md — src/core/policy

## Назначение

Выбор действия (policy / action selection, манифест §3.И). Модуль оценивает
кандидатов-действий по прагматической (близость к предпочитаемому исходу) и
эпистемической (снижение неопределённости) ценности и возвращает решение с
**обязательной причинной трассировкой** (explainability — инвариант).

На S4 policy выбирает **только речевое поведение** (внешние side-effect —
S5/S6). Вход — расширяемый `PolicyContext`: S6 добавляет метакогницию новым
полем с дефолтом (решение A, `stages/S4_SPEC.md`).

См. также:
- `src/core/homeostasis/SPEC.md` — источник гомеостатического отклонения
- `src/speech/SPEC.md` — потребитель (`IntentFrame.goal`)
- ADR-0008 — `IntentFrame.goal` как выход policy; мягкий интент

## Публичный интерфейс

```python
class Action(Enum):
    RESPOND = "respond"
    SILENT = "silent"
    INITIATIVE = "initiative"
    IDENTIFY_PARTNER = "identify_partner"


@dataclass(frozen=True)
class Preferences:
    respond_to_messages: bool = True
    initiative_f_threshold: float = 1.0
    homeostatic_alert: bool = True
    alert_deviation: float = 0.7  # порог тревоги (config, не хардкод)
    silent_baseline: float = 0.5  # базовая ценность SILENT
    silent_stress_gain: float = 0.3  # прирост SILENT со стрессом
    pragmatic_weight: float = 1.0
    epistemic_weight: float = 0.5


@dataclass(frozen=True)
class PolicyContext:
    f: float
    valence: float
    stress: float
    task: str
    homeostasis: HomeostasisState
    has_new_message: bool
    mode: str = "free"  # расширяемое: S6 добавит метакогницию


@dataclass(frozen=True)
class PolicyCandidate:
    action: Action
    pragmatic: float
    epistemic: float
    value: float
    reason: str


@dataclass(frozen=True)
class PolicyTrace:
    chosen: Action
    reason: str
    candidates: tuple[PolicyCandidate, ...]


@dataclass(frozen=True)
class MacroContext:
    task: str = "none"
    mode: str = "free"          # game/cooperative/free


def evaluate_candidates(context, preferences) -> tuple[PolicyCandidate, ...]: ...
def select_action(context, preferences) -> PolicyTrace: ...
```

### Правила оценки (S4)

| Действие | Триггер | Ценность |
|---|---|---|
| `RESPOND` | новое сообщение И `respond_to_messages` | pragmatic=1.0 |
| `SILENT` | всегда (дефолт) | `silent_baseline + silent_stress_gain·conservation(stress)` |
| `INITIATIVE` | `f > initiative_f_threshold` ИЛИ `homeostatic_alert` и отклонение ≥ `alert_deviation` | pragmatic=1.0 |
| `IDENTIFY_PARTNER` | заготовка (драйв — S6) | 0.0 |

`value = pragmatic_weight·pragmatic + epistemic_weight·epistemic`. Тай-брейк —
порядок `Action` (детерминизм). Все пороги — в `Preferences` (CONSTITUTION
§2.2: не хардкодить в логике).

## Инварианты

1. **FC/IS:** `evaluate_candidates`/`select_action` — чистые; одинаковый вход
   → одинаковый `PolicyTrace`.
2. **Explainability:** `reason` выводится из оценки победителя (не постфактум).
3. **Goal-directed:** смена `Preferences` меняет решение без переобучения.
4. **Расширяемость:** `PolicyContext` — frozen; новые поля с дефолтом.
5. **Валидация:** отрицательные пороги/веса → `ValueError`.

## Критерии приёмки

- [x] `evaluate_candidates` — все кандидаты в стабильном порядке
- [x] `select_action` — детерминированный argmax + трасса
- [x] goal-directed (смена `Preferences`)
- [x] explainability (reason + все кандидаты)
- [x] mypy strict, ruff чисты

## Явно НЕ входит

- **Внешние действия (MCP)** — S5/S6.
- **Эпистемический драйв / `IDENTIFY_PARTNER`** — S6.
- **Полный дискретный слой (pymdp)** — S6.
