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
    EXPLORE = "explore"  # S6: эпистемический драйв


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
    identify_threshold: float = 0.7      # S5: порог uncertainty для интента
    partner_trust_floor: float = 0.5     # S5: нижняя граница масштаба RESPOND
    explore_threshold: float = 0.6       # S6: порог неопределённости для EXPLORE


@dataclass(frozen=True)
class PolicyContext:
    f: float
    valence: float
    stress: float
    task: str
    homeostasis: HomeostasisState
    has_new_message: bool
    mode: str = "free"
    partner: PartnerView | None = None  # S5: ToM; None → S4-совместимость
    metacognition: MetacognitionView | None = None  # S6; None → S5-совместимость


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
| `RESPOND` | новое сообщение И `respond_to_messages` | `partner_trust_floor + (1-floor)·trust` (S5) |
| `SILENT` | всегда (дефолт) | `silent_baseline + silent_stress_gain·conservation(stress)` |
| `INITIATIVE` | `f > initiative_f_threshold` ИЛИ `homeostatic_alert` и отклонение ≥ `alert_deviation` | pragmatic=1.0 |
| `IDENTIFY_PARTNER` | `partner.uncertainty ≥ identify_threshold` (S5) | epistemic=1.0 |
| `EXPLORE` | нет сообщения И `metacognition.epistemic_uncertainty ≥ explore_threshold` (S6) | epistemic=1.0 |

`PartnerView` — структурный Protocol (S5): policy не импортирует `tm` (без
цикла); `partner=None` → поведение S4 (RESPOND=1.0, IDENTIFY=0.0).
`MetacognitionView` — структурный Protocol (S6): policy не импортирует
`selfcontrol`; `metacognition=None` → поведение S5 (EXPLORE=0.0). `EXPLORE` —
общий эпистемический драйв; `IDENTIFY_PARTNER` — его агентный частный случай.

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
- **Эпистемический драйв как таковой** — S6 (S5 даёт мягкий интент по
  `uncertainty`, но не самостоятельный исследовательский драйв).
- **Полный дискретный слой (pymdp)** — S6.
