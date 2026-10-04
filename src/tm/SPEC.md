# SPEC.md — src/tm

## Назначение

Теория сознания партнёра (S5, манифест §3.Г). Модуль выводит **сигнатуру
партнёра** из регулярности реплик (стиль/лексика/тайминг/предпочтения) и
ведёт упрощённую ToM: доверие, двусмысленность, конфликт, неопределённость
идентичности. Идентичность — **вывод**, не тег (ADR-0008): поля `speaker_id`
нет, сигнатуры живут в домене общей памяти.

Стадия S5. Полный SPEC стадии — [`stages/S5_SPEC.md`](../../stages/S5_SPEC.md);
план — [`stages/S5_PLAN.md`](../../stages/S5_PLAN.md).

См. также:
- `src/memory/SPEC.md` — эмбеддер/эпизоды/cosine (переиспользуются)
- `src/core/policy/SPEC.md` — потребитель (`PolicyContext.partner`)
- ADR-0008 — атрибуция, узнавание, имена

## Состав

| Файл | Слой | Роль |
|---|---|---|
| `models.py` | Core | `PartnerState`, `PartnerSignature`, `Claim`, `JointGoal` (frozen) |
| `compute.py` | Core | `match_partner`, `update_signature`, `update_trust`, `detect_conflict`, `normalize_pause` (чистые) |
| `partner.py` | Shell | `PartnerModel` — владеет сигнатурами, читает embedder |
| `vigilance.py` | Shell | `VigilanceGate` — утверждения как гипотезы (S5 §4) |
| `joint.py` | Shell | `JointAgency` — совместные цели, подстраховка (S5 §5) |

## Core: сигнатура и матчинг

```python
@dataclass(frozen=True)
class PartnerState:
    trust: float = 0.0          # [0, 1]
    ambiguity: float = 0.0      # [0, 1]
    conflict: float = 0.0       # [0, 1]
    uncertainty: float = 1.0    # [0, 1]
    name: str = ""


@dataclass(frozen=True)
class PartnerSignature:
    centroid: Vector
    weight: float = 1.0
    mean_pause_s: float = 0.0
    mean_valence: float = 0.0
    mean_stress: float = 0.0
    name: str = ""
    aliases: tuple[str, ...] = ()


def match_partner(embedding, signatures, *, threshold) -> tuple[int | None, float]: ...
def update_signature(signature, embedding, *, pause_s, valence, stress,
                     learning_rate) -> PartnerSignature: ...
def update_trust(state, *, conflict, ambiguity, trust_gain,
                 trust_decay) -> PartnerState: ...
```

- **Матчинг:** косинусная близость (`memory.cosine_similarity`), порог,
  тай-брейк — первый максимум. Пустой список → `(None, 0.0)`.
- **Обновление сигнатуры:** скользящее среднее `centroid`/`mean_*`,
  `weight += 1`; вход не мутируется.
- **Доверие:** `trust' = clamp(trust - decay + gain·(1 - conflict))`.

## Shell: PartnerModel

```python
class PartnerModel:
    def __init__(self, embedder, *, match_threshold=0.75, learning_rate=0.2,
                 trust_gain=0.1, trust_decay=0.01) -> None: ...
    def observe(self, text, *, pause_s=0.0, valence=0.0,
                stress=0.0) -> PartnerState: ...
    def set_name(self, name, aliases=()) -> None: ...
    @property
    def signatures(self) -> tuple[PartnerSignature, ...]: ...
    @property
    def state(self) -> PartnerState: ...
```

`observe`: embed → match → обновить/создать сигнатуру → собрать `PartnerState`
(`uncertainty = 1 - similarity`). Сбой эмбеддера → лог + безопасный дефолт
(`uncertainty=1.0`); чат не роняется.

## Shell: VigilanceGate (S5 §4)

```python
class VigilanceGate:
    def __init__(self, embedder, *, conflict_threshold=0.6, max_memory=64) -> None: ...
    def observe(self, text) -> Claim: ...
    def confirm(self) -> Claim | None: ...
    def is_hypothesis(self) -> bool: ...
    @property
    def last_claim(self) -> Claim | None: ...
```

`detect_conflict(claim_embedding, memory_embeddings)` (Core) = `max(1 - cos)`.
Высокий конфликт → низкая `confidence` (гипотеза). **Не блокирует ответ** и не
«обвиняет»: поведение — по сырому сигналу, ярлык — коммуникация (ADR-0008 §1).

## Shell: JointAgency (S5 §5)

```python
class JointAgency:
    def __init__(self, *, default_ttl=20) -> None: ...
    def propose(self, task, *, tick, priority=0.5) -> JointGoal: ...
    def update(self, active_task, *, tick) -> str | None: ...
    def expired(self, tick) -> bool: ...
    def complete(self) -> None: ...
```

Хост подстраховывает: уход от общей цели → напоминание, не исполнение
приказа (манифест §3.Г). TTL снимает заброшенную цель.

## Инварианты

1. **Идентичность — вывод, не тег:** нет `speaker_id` (ADR-0008).
2. **FC/IS:** `compute` — чистые, детерминированные; `partner` — Shell.
3. **ToM не роняет контур:** сбой эмбеддера → лог + дефолт.
4. **Память:** сигнатуры — домен общей памяти; отдельного store нет.
5. **Валидация:** оценки ∈ [0, 1]; `learning_rate` ∈ (0, 1]; `pause_s` ≥ 0.

## Критерии приёмки

- [x] `match_partner` — порог, пустой список, детерминизм
- [x] `update_signature` — сходимость, чистота, границы
- [x] `update_trust` — границы [0, 1]
- [x] `PartnerModel.observe` — сигнатура копится, узнавание
- [x] `set_name` — привязка имени к сигнатуре; вокатив в промпте
- [x] сбой эмбеддера → безопасный дефолт
- [x] `detect_conflict`/`VigilanceGate` — гипотеза по конфликту, без блокировки
- [x] `JointAgency` — совместная цель + напоминание при уходе

## Явно НЕ входит

- **Мультиагентный вход** — S6 (архитектура открыта: множество сигнатур).
- **Отдельное хранилище сигнатур** — не вводится (ADR-0008).
- **Персистентность сигнатур в общий store** — отложено (проход 3/бэклог).
- **Полная sparse-факторизация (pymdp)** — S6.
