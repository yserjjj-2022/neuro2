# SPEC.md — src/core/homeostasis

## Назначение

Гомеостаз интероцептивных каналов (S4): сравнение сырых `severity` сигналов
шины с сетепоинтами и выдача нормированного отклонения. Модуль **отчитывается**
о состоянии — решения принимают policy (речевое предупреждение) и reflex-путь
(throttle). Манифест §3.В, §3.З; ADR-0005 §7.

См. также:
- `src/mcp/SPEC.md` — источник `SignalSource` (severity, is_reflex)
- `src/core/policy/SPEC.md` — потребитель `max_deviation`
- `src/host/throttle.py` — reflex-потребитель `is_critical`

## Публичный интерфейс

```python
@dataclass(frozen=True)
class Setpoint:
    tag: str
    comfort: float = 0.5
    critical: float = 0.9
    weight: float = 1.0
    # валидация: tag непуст, 0 <= comfort < critical <= 1, weight > 0


@dataclass(frozen=True)
class HomeostaticSignal:
    tag: str
    severity: float
    deviation: float
    is_critical: bool


@dataclass(frozen=True)
class HomeostasisState:
    signals: tuple[HomeostaticSignal, ...]
    max_deviation: float
    severity: float
    is_critical: bool


def setpoint_deviation(severity, comfort, critical) -> float:
    """clip((severity - comfort) / (critical - comfort), 0, 1)."""


class Homeostat:
    def __init__(self, setpoints: Sequence[Setpoint], reflex_threshold=0.9) -> None: ...
    def evaluate(self, signals: Sequence[SignalSource]) -> HomeostasisState: ...
```

`critical` согласован с порогом reflex (`severity >= 0.9`): максимальное
отклонение и критический сигнал наступают одновременно.

## Инварианты

1. **FC/IS:** `setpoint_deviation` — чистая; `Homeostat` — Shell (владеет
   сетепоинтами, не мутирует сигналы).
2. **deviation ∈ [0, 1]:** клип; `comfort → 0`, `critical → 1`.
3. **Детерминизм:** порядок результата = порядок сетепоинтов.
4. **Каналы без сетепоинта игнорируются.**
5. **Fail-fast:** `critical <= comfort`, `comfort/critical ∉ [0, 1]`,
   `weight <= 0`, пустой список → `ValueError`.

## Критерии приёмки

- [x] `setpoint_deviation` — чистая, границы/линейность
- [x] `Setpoint` — валидация границ
- [x] `Homeostat.evaluate` — порядок, отсутствие канала, агрегаты
- [x] `is_critical` при `severity >= reflex_threshold`
- [x] mypy strict, ruff чисты

## Явно НЕ входит

- **Корневые приоры** (когерентность/целостность/верификация) — S5/S6.
- **Принятие решений** — policy/reflex (вне модуля).
- **Гистерезис тревоги** — Open Question S4 (калибровка по телеметрии).
