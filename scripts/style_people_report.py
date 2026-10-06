"""Component-feasibility on REAL people (S7 discussion).

Question: does the *style* channel carry identity across unseen topics/categories
for real participants, and which FEATURE FAMILY carries it — beyond semantic
embeddings?

Input: ``data/style_study/answers.jsonl`` (participants × stimuli, from the
offline collector). Same track for everyone.

Families (ablation):
    lex    — function-word frequencies
    punct  — punctuation / paralinguistic markers (!!!, ..., )), caps, repeats)
    struct — length, word counts, TTR, mean word length
    affect — expressivity profile (exclamations, caps, emoji, laughter)
    timing — latency_ms
    embed  — semantic embedding (baseline: "does it just catch the topic?")
    style  — lex + punct + struct
    all    — style + affect + timing

Metrics (per signal):
    within/between distance, separation, rank-1 LOO, silhouette,
    cross-category (leave-one-category-out) — the key test for identity.

Decision rule (pre-registered):
    a family is "identity-bearing" if cross-category rank-1 is clearly above
    the 1/n_participants chance (~0.33 for 3) AND above the embedding baseline.

Output: printed report. No files written (read-only analysis).

Run: uv run python scripts/style_people_report.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Запуск из scripts/: корень репозитория — на уровень выше.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from style_feasibility import (
    EMOJI_RE,
    FUNCTION_WORDS,
    PUNCT,
    WORD_RE,
    cosine_distances,
    cross_topic_accuracy,
    pairwise_distances,
    rank1_loo,
    silhouette,
    within_between,
)

from src.memory.embedder import build_embedder

DATA = Path("data/style_study/answers.jsonl")

# Паралингвистические маркеры: повтор пунктуации, скобки-смайлы, смех, КАПС-слова.
LAUGH_RE = re.compile(r"(а+х+а+|хе+х+|ха-?ха|лол|\){2,})", re.IGNORECASE)
REPEAT_RE = re.compile(r"(.)\1{2,}")  # три и более одинаковых символа подряд
CAPS_WORD_RE = re.compile(r"[А-ЯЁA-Z]{2,}")


def normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


# ---------------------------------------------------------------------------
# Feature families
# ---------------------------------------------------------------------------


def lex_features(text: str) -> list[float]:
    words = WORD_RE.findall(normalize(text))
    n = max(1, len(words))
    counts = Counter(words)
    return [counts.get(fw, 0) / n for fw in FUNCTION_WORDS]


def punct_features(text: str) -> list[float]:
    words = WORD_RE.findall(normalize(text))
    n = max(1, len(words))
    feats = [text.count(p) / n for p in PUNCT]
    feats.append(text.count("))") / n)          # скобка-смайл
    feats.append(text.count("((") / n)
    feats.append(len(REPEAT_RE.findall(text)) / n)  # привееет
    feats.append(len(LAUGH_RE.findall(text)) / n)   # ахаха
    feats.append(text.count("...") / n)         # многоточие явно
    feats.append(text.count("\n") / n)
    return feats


def struct_features(text: str) -> list[float]:
    words = WORD_RE.findall(normalize(text))
    n = max(1, len(words))
    return [
        len(text) / 50.0,                       # длина сообщения (масштаб)
        n / 10.0,                               # число слов
        float(np.mean([len(w) for w in words])) if words else 0.0,
        len(set(words)) / n,                    # TTR
        (text.count(".") + text.count("!") + text.count("?") + text.count("…")),
    ]


def affect_features(text: str) -> list[float]:
    words = WORD_RE.findall(normalize(text))
    n = max(1, len(words))
    n_chars = max(1, len(text))
    return [
        text.count("!") / n,                    # возбуждение
        sum(1 for c in text if c.isupper()) / n_chars,   # КАПС
        len(EMOJI_RE.findall(text)) / n,         # эмодзи
        len(LAUGH_RE.findall(text)) / n,         # смех
        text.count("?") / n,                    # вопросительность
    ]


def timing_features(latency_ms: float) -> list[float]:
    # log-масштаб: латентности тяжелохвостые.
    return [float(np.log1p(max(0.0, latency_ms)) / 10.0)]


def len_features(text: str) -> list[float]:
    # Контроль: только длина (в символах). Проверяем, не «стиль» ли это на самом деле.
    return [len(text) / 50.0]


FAMILIES = {
    "lex": lex_features,
    "punct": punct_features,
    "struct": struct_features,
    "affect": affect_features,
    "length": len_features,
}


def build_matrix(records: list[dict], family: str) -> np.ndarray:
    rows: list[list[float]] = []
    for r in records:
        if family == "timing":
            rows.append(timing_features(r.get("latency_ms", 0.0)))
        else:
            rows.append(FAMILIES[family](r["text"]))
    return np.asarray(rows, dtype=np.float64)


def zscore(matrix: np.ndarray) -> np.ndarray:
    mu = matrix.mean(axis=0)
    sd = matrix.std(axis=0)
    return (matrix - mu) / np.where(sd == 0.0, 1.0, sd)


def combine(*families: str, records: list[dict]) -> np.ndarray:
    parts = [zscore(build_matrix(records, f)) for f in families]
    return np.hstack(parts)


def evaluate(dist: np.ndarray, labels: list[str], cats: list[str]) -> dict[str, float]:
    within, between = within_between(dist, labels)
    return {
        "within": within,
        "between": between,
        "sep": (between - within) / between if between > 0 else 0.0,
        "rank1": rank1_loo(dist, labels),
        "sil": silhouette(dist, labels),
        "xcat": cross_topic_accuracy(dist, labels, cats),
    }


def permutation_pvalue(
    dist: np.ndarray, labels: list[str], cats: list[str], *, n_perm: int = 1000
) -> float:
    """Перестановочный тест для xcat: доля перемешиваний не хуже наблюдённого.

    Метки участников перемешиваются ВНУТРИ категорий (сохраняем структуру тем),
    чтобы проверить: даёт ли сигнал идентичность сверх случайного.
    """
    observed = cross_topic_accuracy(dist, labels, cats)
    labels_arr = np.asarray(labels)
    cats_arr = np.asarray(cats)
    rng = np.random.default_rng(0)
    ge = 0
    for _ in range(n_perm):
        perm = labels_arr.copy()
        for cat in dict.fromkeys(cats):
            idx = np.where(cats_arr == cat)[0]
            perm[idx] = rng.permutation(perm[idx])
        if cross_topic_accuracy(dist, list(perm), cats) >= observed:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def print_block(title: str, rows: dict[str, dict[str, float]], chance: float) -> None:
    print(f"\n=== {title} (шанс ≈ {chance:.3f}) ===")
    print(
        f"{'signal':12s} {'within':>8s} {'between':>8s} {'sep':>7s} "
        f"{'rank1':>7s} {'sil':>7s} {'xcat':>7s}"
    )
    for name, m in rows.items():
        print(
            f"{name:12s} {m['within']:8.3f} {m['between']:8.3f} {m['sep']:7.3f} "
            f"{m['rank1']:7.3f} {m['sil']:7.3f} {m['xcat']:7.3f}"
        )


def main() -> None:
    load_dotenv()

    records = [
        json.loads(line)
        for line in DATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    participants = [r["participant_id"] for r in records]
    categories = [r["category"] for r in records]
    n_part = len(set(participants))
    chance = 1.0 / n_part

    print(
        f"Записей: {len(records)} | участников: {n_part} "
        f"({sorted(set(participants))}) | категорий: {len(set(categories))}"
    )

    # Семейства по отдельности + комбинации.
    dist: dict[str, np.ndarray] = {}
    for fam in list(FAMILIES) + ["timing"]:
        dist[fam] = pairwise_distances(zscore(build_matrix(records, fam)))

    dist["style"] = pairwise_distances(combine("lex", "punct", "struct", records=records))
    dist["style+aff"] = pairwise_distances(
        combine("lex", "punct", "struct", "affect", records=records)
    )
    dist["all"] = pairwise_distances(
        combine("lex", "punct", "struct", "affect", "timing", records=records)
    )

    # Эмбеддинги — семантический бейзлайн.
    embedder = build_embedder("auto")
    emb = np.vstack([embedder.embed(r["text"]) for r in records])
    dist["embed"] = cosine_distances(emb)

    person_rows = {name: evaluate(d, participants, categories) for name, d in dist.items()}
    print_block("Разделимость УЧАСТНИКОВ", person_rows, chance)

    # Утечка темы/категории: насколько сигнал ловит категорию, а не человека.
    cat_rows = {name: evaluate(d, categories, categories) for name, d in dist.items()}
    print_block("Разделимость КАТЕГОРИЙ (утечка темы)", cat_rows, 1.0 / len(set(categories)))

    print("\n=== ВЕРДИКТ ===")
    emb_x = person_rows["embed"]["xcat"]
    print(f"embed xcat (бейзлайн): {emb_x:.3f} | rank1={person_rows['embed']['rank1']:.3f}")
    print(f"контроль 'только длина': xcat={person_rows['length']['xcat']:.3f}")
    print("\nперестановочный тест (p, меньше = значимее):")
    for name in ["lex", "punct", "struct", "affect", "timing", "length", "style", "all", "embed"]:
        p = permutation_pvalue(dist[name], participants, categories)
        print(f"  {name:10s} xcat={person_rows[name]['xcat']:.3f}  p={p:.3f}")


if __name__ == "__main__":
    main()
