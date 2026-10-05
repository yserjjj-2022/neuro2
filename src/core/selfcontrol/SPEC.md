# SPEC.md — src/core/selfcontrol

## Назначение

Самоконтроль first-class (S6, манифест §6.4, INTENT §4): метакогнитивные
наблюдаемые, детектор тихого дрейфа (critical slowing down), классификатор
«развитие vs дрейф» и протокол сброса. Functional Core / Imperative Shell
(ADR-0004). Решения — ADR-0009.

## Публичный интерфейс

### Core (models.py)

```python
@dataclass(frozen=True)
class Metacognition:
    conflict: float              # несогласие ансамбля, [0, 1]
    metastability: float         # частота смен аттрактора, [0, 1]
    epistemic_uncertainty: float # неопределённость, [0, 1]
    saturation: float            # насыщение/тренд F, [0, 1]

@dataclass(frozen=True)
class CriticalSlowingDown:
    variance: float
    autocorrelation: float
    slowing: float               # [0, 1]
    is_warning: bool

class ChangeKind(Enum): STABLE | DEVELOPMENT | DRIFT

@dataclass(frozen=True)
class ChangeAssessment:
    kind: ChangeKind
    core_preserved: bool
    traceable: bool
    coherent: bool
    reason: str

class ResetLevel(Enum): SOFT | FREEZE | HARD

@dataclass(frozen=True)
class ResetPlan:
    level: ResetLevel
    reason: str
    triggered: bool
```

### Core (compute.py)

```python
def compute_conflict(scores) -> float: ...                 # энтропия/норм., [0,1]
def compute_metastability(switch_flags, *, window) -> float: ...
def compute_saturation(f_trend, *, threshold=None) -> float: ...
def critical_slowing_down(series, *, variance_gain, autocorr_gain) -> CriticalSlowingDown: ...
def classify_change(*, core_preserved, traceable, coherent, changed=True) -> ChangeAssessment: ...
def plan_reset(*, slowing, assessment, soft_threshold=0.5, hard_core_broken=False) -> ResetPlan: ...
```

### Shell (monitor.py)

```python
class SelfMonitor:
    def __init__(self, *, window=50, variance_gain=1.0, autocorr_gain=1.0,
                 warning_threshold=0.6, soft_threshold=0.5) -> None: ...
    def observe(self, *, scores, switched, f, partner_uncertainty,
                core_preserved=True, traceable=True, coherent=True
                ) -> tuple[Metacognition, ResetPlan]: ...
    @property
    def metacognition(self) -> Metacognition | None: ...
    @property
    def last_assessment(self) -> ChangeAssessment | None: ...
```

## Семантика решений (ADR-0009)

- **Наблюдаемые не дублируют аффект** (F/valence/stress/γ): conflict — из
  scores, metastability — из смен маски, saturation — из тренда F.
- **Триггер — не стресс:** сброс по critical slowing down (дисперсия +
  lag-1 автокорреляция) в связке с классификацией.
- **Три механизма:** SOFT (adaptive reset), FREEZE (regime shift), HARD
  (catastrophic drift). Core не сбрасывается.
- **Read-only:** наблюдаемые влияют на следующий тик/ход, не на F того же.

## Самоотчёт сброса (S7-D)

`report.py` — чистый Core: наблюдаемый сброс → структурированный честный отчёт.

```python
@dataclass(frozen=True)
class ResetReport:
    level: str        # "", soft/freeze/hard
    change_kind: str  # "", stable/development/drift
    triggered: bool
    certain: bool     # False → неизвестный уровень (не выдумываем)
    text: str
    reason: str

def reset_self_report(*, reset_level, change_kind=None, reason="") -> ResetReport: ...
```

Правила честности: сброса не было → явно сказать; известный уровень → описание
уровня + классификация изменения; неизвестный уровень → `certain=False`
(детали не фабрикуются). Речевой интент `report_reset` и исполнение через
`CapabilityGate` — в `src/speech`.

## Инварианты

1. Все наблюдаемые ∈ [0, 1]; валидация в `__post_init__`.
2. Детерминизм: одинаковый вход → одинаковый снимок/план.
3. Короткий ряд → безопасный дефолт (slowing=0.0, не warning).
4. `changed=False` → STABLE.
5. Core нарушен → HARD (никогда не «сброс к норме»).
6. LLM не участвует (ADR-0007).

## Критерии приёмки (S6, проход 1)

- [ ] наблюдаемые считаются и не дублируют аффект
- [ ] CSD ловит рост дисперсии/автокорреляции
- [ ] классификатор различает развитие/дрейф по трём осям
- [ ] протокол сброса: три уровня, core не сбрасывается
- [ ] Shell копит окно и отдаёт наблюдаемые + план
- [ ] тесты/ruff зелёные

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Формула CSD | Открыто | `sigmoid(z_var·g + z_ac·g)`; калибровка по C10 |
| Пороги классификатора | Открыто | калибровка по телеметрии |
| Канал шины для метакогниции | Отложено | проход 2 |
| Расписание ночного сна | Отложено | проход 2 |
