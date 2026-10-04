# SPEC.md — src/core/factorization

## Назначение

Разреженный дискретный слой (манифест §3.Е): вместо одной комбинаторной
POMDP-матрицы — произведение независимых малых факторов (`mode` / `partner` /
`task`). Совместное распределение не материализуется, обновление —
покомпонентное. Functional Core (ADR-0004); NumPy-only (`pymdp` — опциональное
будущее, ADR-0009 §6). Решения — ADR-0005 §3.

## Публичный интерфейс

### models.py

```python
@dataclass(frozen=True)
class Factor:
    name: str
    states: tuple[str, ...]      # уникальные
    prior: Vector                # норм., shape=(n,)
    likelihood: Vector           # норм., shape=(n,)

@dataclass(frozen=True)
class FactorizedState:
    factors: tuple[Factor, ...]  # уникальные имена
```

### compute.py

```python
def posterior(factor) -> Vector: ...                       # prior ∘ likelihood → норм.
def marginal(state, name) -> Vector: ...                   # распределение фактора
def update_factor(state, name, observation, *, learning_rate=1.0) -> FactorizedState: ...
def argmax_state(factor) -> str: ...                       # MAP
```

## Семантика

- **Независимость:** `update_factor` меняет только целевой фактор; остальные
  переносятся как есть.
- **Нормировка:** при нулевой сумме — равномерное распределение.
- **Смесь:** новый приор = `(1-lr)·posterior + lr·normalize(observation)`.
- **Детерминизм:** одинаковый вход → одинаковый выход.

## Инварианты

1. `prior`/`likelihood` неотрицательны, размерность == числу состояний.
2. `states` и имена факторов уникальны.
3. `posterior` нормирован (сумма 1).
4. `update_factor` не мутирует вход (frozen).
5. `learning_rate ∈ (0, 1]`.
6. LLM не участвует (ADR-0007).

## Критерии приёмки (S6, проход 1)

- [ ] факторы независимы, маргинал по имени
- [ ] posterior нормирован
- [ ] обновление изолировано (другие факторы не меняются)
- [ ] MAP-состояние
- [ ] валидация + детерминизм
- [ ] тесты/ruff зелёные

## Open Questions

| Вопрос | Статус | Решение |
|---|---|---|
| Набор факторов | Открыто | mode/partner/task в проходе 1; расширение — по мере надобности |
| `pymdp` | Отложено | NumPy-факторизация (ADR-0009 §6) |
| Связь с policy (управление режимом) | Отложено | проход 2 |
