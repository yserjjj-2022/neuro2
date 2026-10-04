"""Domain objects for the theory-of-mind module (S5).

Frozen dataclasses: immutable snapshots, analogous to ``PolicyContext`` /
``HomeostasisState``. Identity is a *derived regularity*, not a tag (ADR-0008):
``PartnerSignature`` accumulates the centroid of a partner's utterances, not a
``speaker_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.memory.serialize import Vector


@dataclass(frozen=True)
class PartnerState:
    """Снимок состояния партнёра (упрощённая ToM, манифест §3.Г).

    Attributes:
        trust: Доверие к партнёру, [0, 1] (оценка согласия).
        ambiguity: Двусмысленность последних реплик, [0, 1].
        conflict: Конфликт с накопленным, [0, 1].
        uncertainty: Неопределённость идентичности, [0, 1]
            (насколько плохо матчится сигнатура).
        name: Принятое имя (символ-якорь) или "" (не объявлено).
    """

    trust: float = 0.0
    ambiguity: float = 0.0
    conflict: float = 0.0
    uncertainty: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        """Валидация: все оценки в [0, 1].

        Raises:
            ValueError: Если любое из trust/ambiguity/conflict/uncertainty
                вне [0, 1].
        """
        for field_name in ("trust", "ambiguity", "conflict", "uncertainty"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class Claim:
    """Утверждение партнёра как гипотеза (Vigilance Gate, S5 §4).

    Новое утверждение маркируется как гипотеза, пока не подтвердится
    практикой; защита от эпистемического дрейфа (манифест §3.Г).

    Attributes:
        content: Текст утверждения.
        confidence: Уверенность в гипотезе, [0, 1] (растёт с подтверждением).
        conflict: Рассогласование с накопленным, [0, 1].
        confirmed: Подтверждено ли практикой.
    """

    content: str
    confidence: float
    conflict: float
    confirmed: bool = False

    def __post_init__(self) -> None:
        for field_name in ("confidence", "conflict"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class JointGoal:
    """Совместная цель хоста и партнёра (Joint Agency, S5 §5).

    Хост подстраховывает, а не только исполняет: общая цель имеет TTL и
    приоритет; при уходе партнёра от общей цели — реакция (напоминание), а
    не исполнение приказа (манифест §3.Г).

    Attributes:
        task: Тег общей задачи/аттрактора.
        ttl_ticks: Сколько тиков цель удерживается без подтверждения.
        priority: Приоритет цели, [0, 1].
        created_tick: Тик создания (для расчёта возраста).
    """

    task: str
    ttl_ticks: int = 20
    priority: float = 0.5
    created_tick: int = 0

    def __post_init__(self) -> None:
        if self.ttl_ticks < 1:
            raise ValueError(f"ttl_ticks must be >= 1, got {self.ttl_ticks}")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError(f"priority must be in [0, 1], got {self.priority}")


@dataclass(frozen=True)
class PartnerSignature:
    """Сигнатура партнёра — выведенная регулярность (не тег, ADR-0008).

    Attributes:
        centroid: Усреднённый эмбеддинг реплик партнёра.
        weight: Накопленный вес (число/значимость взаимодействий).
        mean_pause_s: Средний интервал между репликами, с.
        mean_valence: Средняя реакция хоста (предпочтения).
        mean_stress: Средний стресс в диалоге.
        name: Объявленное имя ("" если нет).
        aliases: Объявленные алиасы.
    """

    centroid: Vector
    weight: float = 1.0
    mean_pause_s: float = 0.0
    mean_valence: float = 0.0
    mean_stress: float = 0.0
    name: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)
