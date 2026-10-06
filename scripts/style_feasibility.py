"""Style-channel feasibility probe (throwaway experiment, S7 discussion).

Question: can a *stylometric* vector (function-word frequencies + punctuation +
structure) separate personas on SHORT Russian replies better than semantic
embeddings do?

Method:
    personas × topics × reps -> real LLM replies (persona system prompt, fixed
    user question per topic). Each reply gets (a) a style vector, (b) an
    embedding. We compare within-persona vs between-persona distances, rank-1
    leave-one-out accuracy and silhouette on BOTH signals.

Decision rule (pre-registered):
    style rank-1 >= 0.8 AND silhouette >= 0.3 AND style >= embedding
        -> style channel is viable for S5 identity
    else
        -> not viable; keep enrollment + topic continuity

Output: reports/replies.jsonl (raw transcript) + a printed report.

Run: uv run python scripts/style_feasibility.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.memory.embedder import build_embedder
from src.speech.llm import build_llm_client

# ---------------------------------------------------------------------------
# Scenario data
# ---------------------------------------------------------------------------

PERSONAS: dict[str, str] = {
    "dry": (
        "Ты пишешь короткие сообщения в мессенджере. Стиль: сухой, отрывистый, "
        "почти без знаков препинания, всё строчными буквами, без эмодзи, "
        "без вводных слов. Максимум 6-8 слов. Никаких приветствий и вежливости."
    ),
    "warm": (
        "Ты пишешь сообщения в мессенджере. Стиль: тёплый, ласковый, много "
        "эмодзи, восклицательные знаки, уменьшительно-ласкательные слова "
        "(солнышко, обнимаю, милый). Пиши 1-2 короткие фразы."
    ),
    "formal": (
        "Ты пишешь сообщения в мессенджере. Стиль: официально-деловой, длинные "
        "полные предложения, вводные обороты (однако, таким образом, в рамках, "
        "в связи с этим), точки с запятой, без эмодзи и без сленга."
    ),
    "chaotic": (
        "Ты пишешь сообщения в мессенджере. Стиль: хаотичный, рваный, много "
        "многоточий, тире, обрывков фраз, сленга и слов-паразитов (ну, типа, "
        "вообще, короче), иногда КАПСОМ. Без эмодзи."
    ),
}

TOPICS: list[tuple[str, str]] = [
    ("weather", "Как погода у тебя сегодня?"),
    ("dinner", "Что посоветуешь на ужин?"),
    ("work", "Как дела на работе?"),
    ("weekend", "Чем займёшься на выходных?"),
    ("dog", "Стоит ли завести собаку?"),
]

REPS = 3
TEMPERATURE = 0.9

# Русские служебные слова (частотный список; темо-независимый сигнал).
FUNCTION_WORDS: tuple[str, ...] = tuple(
    dict.fromkeys(
        """
        и в во не что он на я с со как а то все она так его но да ты к ко у же вы
        за бы по только ее мне было вот от меня еще нет о об из ему теперь когда
        даже ну вдруг ли если уже или ни был была были быть до вас нам уж ведь там
        потом себя ей может они тут где есть надо для мы тебя их чем сам чтоб чтобы
        без будто чего раз тоже себе под будет тогда кто этот того потому этого
        какой совсем здесь этом один почти мой тем нее сейчас куда зачем всех
        никогда можно при наконец два другой хоть после над больше тот через эти
        нас про всего них какая много разве три эту моя впрочем свою этой перед
        иногда лучше чуть том нельзя такой им более всегда конечно всю между
        """.split()  # noqa: SIM905 - читаемый словарь важнее литерала
    )
)

# Знаки/символы, различающие стиль.
PUNCT = (",", ".", "!", "?", "…", "—", "-", ";", ":", "(", ")", '"', "*")
EMOJI_RE = re.compile("[\U0001f300-\U0001faff\u2600-\u27bf\u2764]")
WORD_RE = re.compile(r"[а-яёa-z]+")


def normalize(text: str) -> str:
    """Нижний регистр + ё→е (устойчивость к орфографическому разнобою)."""
    return text.lower().replace("ё", "е")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def style_features(text: str) -> tuple[np.ndarray, list[str]]:
    """Стилевой вектор реплики: служебные слова + пунктуация + структура.

    Returns:
        (vector, feature_names)
    """
    low = normalize(text)
    words = WORD_RE.findall(low)
    n_words = max(1, len(words))
    n_chars = max(1, len(text))

    features: list[float] = []
    names: list[str] = []

    # 1. Относительные частоты служебных слов.
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    for fw in FUNCTION_WORDS:
        features.append(counts.get(fw, 0) / n_words)
        names.append(f"fw:{fw}")

    # 2. Пунктуация (на слово).
    for p in PUNCT:
        features.append(text.count(p) / n_words)
        names.append(f"punct:{p}")

    # 3. Структура.
    features.append(len(words) / 10.0)
    names.append("struct:n_words")
    features.append(float(np.mean([len(w) for w in words])) if words else 0.0)
    names.append("struct:mean_word_len")
    features.append(len(set(words)) / n_words)
    names.append("struct:ttr")
    features.append(sum(1 for c in text if c.isupper()) / n_chars)
    names.append("struct:upper_ratio")
    features.append(len(EMOJI_RE.findall(text)) / n_words)
    names.append("struct:emoji_rate")
    features.append(text.count("\n") / n_words)
    names.append("struct:newlines")

    return np.asarray(features, dtype=np.float64), names


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    """Матрица евклидовых расстояний между строками."""
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def cosine_distances(matrix: np.ndarray) -> np.ndarray:
    """Матрица косинусных расстояний (1 - cos); строки L2-нормированы."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.where(norms == 0.0, 1.0, norms)
    return 1.0 - unit @ unit.T


def within_between(dist: np.ndarray, labels: list[str]) -> tuple[float, float]:
    """Средняя внутри- и между-классовая дистанция."""
    labels_arr = np.asarray(labels)
    same = labels_arr[:, None] == labels_arr[None, :]
    np.fill_diagonal(same, False)
    within = float(dist[same].mean())
    between = float(dist[~same].mean())
    return within, between


def rank1_loo(dist: np.ndarray, labels: list[str]) -> float:
    """Leave-one-out точность ближайшего соседа (та же метка?)."""
    n = dist.shape[0]
    correct = 0
    for i in range(n):
        order = np.argsort(dist[i])
        nearest = next(j for j in order if j != i)
        correct += int(labels[nearest] == labels[i])
    return correct / n


def cross_topic_accuracy(
    dist: np.ndarray, labels: list[str], topics: list[str]
) -> float:
    """Точность идентификации на НЕВИДАННОЙ теме (leave-one-topic-out).

    Для каждой темы-холдаут ближайший сосед ищется только среди реплик
    остальных тем. Если сигнал темо-инвариантен — точность высокая.
    """
    labels_arr = np.asarray(labels)
    topics_arr = np.asarray(topics)
    correct = 0
    total = 0
    for topic in dict.fromkeys(topics):
        test = np.where(topics_arr == topic)[0]
        train = np.where(topics_arr != topic)[0]
        for i in test:
            nearest = train[np.argmin(dist[i, train])]
            correct += int(labels_arr[nearest] == labels_arr[i])
            total += 1
    return correct / total if total else 0.0


def pca(matrix: np.ndarray, k: int) -> np.ndarray:
    """Проекция на первые k главных компонент (центрирование внутри)."""
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:k].T


def silhouette(dist: np.ndarray, labels: list[str]) -> float:
    """Средний силуэт по предвычисленной матрице расстояний."""
    labels_arr = np.asarray(labels)
    unique = list(dict.fromkeys(labels))
    scores: list[float] = []
    for i in range(dist.shape[0]):
        same = [j for j in range(dist.shape[0]) if j != i and labels_arr[j] == labels_arr[i]]
        a = float(dist[i, same].mean()) if same else 0.0
        b = math.inf
        for other in unique:
            if other == labels_arr[i]:
                continue
            members = [j for j in range(dist.shape[0]) if labels_arr[j] == other]
            if members:
                b = min(b, float(dist[i, members].mean()))
        if math.isinf(b):
            continue
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass
class Reply:
    persona: str
    topic: str
    text: str


def generate_replies(llm) -> list[Reply]:  # type: ignore[no-untyped-def]
    """Сгенерировать реплики: персоны × темы × повторы."""
    replies: list[Reply] = []
    for persona, style in PERSONAS.items():
        for topic, question in TOPICS:
            for rep in range(REPS):
                messages = [
                    {"role": "system", "content": style},
                    {"role": "user", "content": question},
                ]
                try:
                    text = llm.reply(messages, max_tokens=80)
                except Exception as exc:  # noqa: BLE001 - эксперимент, логируем и идём дальше
                    text = f"[ERROR {exc}]"
                replies.append(Reply(persona=persona, topic=topic, text=text))
                print(f"  {persona:8s} {topic:8s} #{rep}: {text[:60]}")
    return replies


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def evaluate(
    dist: np.ndarray, labels: list[str], topics: list[str]
) -> dict[str, float]:
    within, between = within_between(dist, labels)
    return {
        "within": within,
        "between": between,
        "separation": (between - within) / between if between > 0 else 0.0,
        "rank1": rank1_loo(dist, labels),
        "silhouette": silhouette(dist, labels),
        "cross_topic": cross_topic_accuracy(dist, labels, topics),
    }


def load_replies(path: Path) -> list[Reply]:
    """Прочитать ранее сохранённые реплики (для повторного анализа)."""
    return [
        Reply(persona=d["persona"], topic=d["topic"], text=d["text"])
        for d in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    ]


def print_block(title: str, rows: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {title} ===")
    print(
        f"{'signal':12s} {'within':>8s} {'between':>8s} {'sep':>7s} "
        f"{'rank1':>7s} {'silhou':>8s} {'xtopic':>7s}"
    )
    for name, m in rows.items():
        print(
            f"{name:12s} {m['within']:8.3f} {m['between']:8.3f} "
            f"{m['separation']:7.3f} {m['rank1']:7.3f} {m['silhouette']:8.3f} "
            f"{m['cross_topic']:7.3f}"
        )


def main() -> None:
    load_dotenv()
    out = Path("reports/replies.jsonl")
    fresh = "--fresh" in sys.argv or not out.exists()

    if fresh:
        llm = build_llm_client("auto")
        print(f"LLM: {type(llm).__name__}\nГенерация реплик...")
        replies = generate_replies(llm)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for reply in replies:
                handle.write(
                    json.dumps(
                        {
                            "persona": reply.persona,
                            "topic": reply.topic,
                            "text": reply.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    else:
        print(f"LLM: (reuse) {out}")
        replies = load_replies(out)

    embedder = build_embedder("auto")
    texts = [r.text for r in replies]
    personas = [r.persona for r in replies]
    topics = [r.topic for r in replies]

    # Стилевые признаки → z-стандартизация (+ PCA-вариант).
    style_matrix = np.vstack([style_features(t)[0] for t in texts])
    mu = style_matrix.mean(axis=0)
    sd = style_matrix.std(axis=0)
    style_z = (style_matrix - mu) / np.where(sd == 0.0, 1.0, sd)
    style_pca = pca(style_z, 10)

    emb_matrix = np.vstack([embedder.embed(t) for t in texts])

    style_dist = pairwise_distances(style_z)
    style_pca_dist = pairwise_distances(style_pca)
    emb_dist = cosine_distances(emb_matrix)

    print_block(
        "Разделимость ПЕРСОН",
        {
            "style": evaluate(style_dist, personas, topics),
            "style+pca10": evaluate(style_pca_dist, personas, topics),
            "embed": evaluate(emb_dist, personas, topics),
        },
    )
    print_block(
        "Разделимость ТЕМ (утечка)",
        {
            "style": evaluate(style_dist, topics, topics),
            "style+pca10": evaluate(style_pca_dist, topics, topics),
            "embed": evaluate(emb_dist, topics, topics),
        },
    )

    sp = evaluate(style_pca_dist, personas, topics)
    ep = evaluate(emb_dist, personas, topics)
    viable = sp["rank1"] >= 0.8 and sp["silhouette"] >= 0.3 and sp["cross_topic"] >= 0.8
    print("\n=== ВЕРДИКТ ===")
    print(
        f"style+pca10: rank1={sp['rank1']:.3f} sil={sp['silhouette']:.3f} "
        f"xtopic={sp['cross_topic']:.3f} | "
        f"embed: rank1={ep['rank1']:.3f} sil={ep['silhouette']:.3f} "
        f"xtopic={ep['cross_topic']:.3f}"
    )
    print(
        "СТИЛЕВОЙ КАНАЛ: "
        + ("ЖИЗНЕСПОСОБЕН" if viable else "НЕ подтверждён в этом эксперименте")
    )
    print(f"Транскрипт: {out}")


if __name__ == "__main__":
    main()
