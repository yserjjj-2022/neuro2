"""Import collected answers (from the offline HTML collector) into JSONL.

Reads ``answers_<ID>.json`` files (downloaded by colleagues) and merges them
into ``data/style_study/answers.jsonl`` with de-duplication on
``(participant_id, stimulus_id)`` (a later submission wins).

Run:
    uv run python scripts/import_answers.py data/style_study/answers_P1.json ...
    uv run python scripts/import_answers.py            # scan the whole dir
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_DIR = Path("data/style_study")
REQUIRED = ("participant_id", "stimulus_id", "text")


def read_answers(path: Path) -> list[dict]:
    """Прочитать один файл ответов.

    Поддерживает два формата:
        * новый объект ``{"schema_version", "participant_id", "answers": [...]}``;
        * старый массив записей ``[{...}, ...]`` (обратная совместимость).

    Args:
        path: Путь к answers_*.json.

    Returns:
        Список записей-словарей.

    Raises:
        TypeError: Если структура не объект и не массив.
        ValueError: Если у записи нет обязательных полей.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = data.get("answers")
        if not isinstance(records, list):
            raise TypeError(f"{path.name}: 'answers' must be a list")
    elif isinstance(data, list):
        records = data
    else:
        raise TypeError(f"{path.name}: expected a JSON object or array")
    for record in records:
        missing = [k for k in REQUIRED if k not in record or not str(record[k]).strip()]
        if missing:
            raise ValueError(f"{path.name}: record missing {missing}: {record!r}")
    return records


def load_jsonl(path: Path) -> dict[tuple[str, str], dict]:
    """Прочитать существующий answers.jsonl в словарь по ключу (участник, стимул)."""
    if not path.exists():
        return {}
    merged: dict[tuple[str, str], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        merged[(record["participant_id"], record["stimulus_id"])] = record
    return merged


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]]
    if args:
        files = args
    else:
        files = sorted(DEFAULT_DIR.glob("answers_*.json"))
    if not files:
        print(f"no answers_*.json found in {DEFAULT_DIR}")
        return

    out = DEFAULT_DIR / "answers.jsonl"
    merged = load_jsonl(out)
    before = len(merged)
    for path in files:
        records = read_answers(path)
        for record in records:
            merged[(record["participant_id"], record["stimulus_id"])] = record
        participants = {r["participant_id"] for r in records}
        print(f"{path.name}: +{len(records)} записей от {sorted(participants)}")

    ordered = sorted(
        merged.values(),
        key=lambda r: (r["participant_id"], r.get("order", 0), r["stimulus_id"]),
    )
    with out.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    per_participant: dict[str, int] = {}
    for record in merged.values():
        per_participant[record["participant_id"]] = (
            per_participant.get(record["participant_id"], 0) + 1
        )
    print(f"\n{out}: {before} → {len(merged)} записей")
    for pid in sorted(per_participant):
        print(f"  {pid}: {per_participant[pid]} ответов")


if __name__ == "__main__":
    main()
