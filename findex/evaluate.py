"""precision@5 для TF-IDF vs BM25 на розмічених запитах.

eval_queries.json:
  [{"query": "python AND async", "relevant": ["назва_файлу_без_txt", "інша"]}, ...]

Запуск: python -m findex.evaluate index.bin eval_queries.json
"""

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict, cast

from .model import Index, IndexFormat, open_index
from .scoring import BM25, Scorer, TfIdf
from .search import Matcher, SearchResult, search


class LabelledQuery(TypedDict):
    query: str
    relevant: list[str]


@dataclass(frozen=True)
class EvalRow:
    query: str
    scores: dict[str, float]


def precision_at_k(
    results: Sequence[SearchResult], relevant_titles: Iterable[str], k: int = 5
) -> float:
    """Скільки з топ-k результатів розмічено як правильні, поділити на k."""
    relevant = set(relevant_titles)
    return sum(1 for r in results[:k] if r.title in relevant) / k


def evaluate(
    index: Index,
    labelled: Sequence[LabelledQuery],
    scorers: Mapping[str, Scorer],
    k: int = 5,
) -> list[EvalRow]:
    matcher = Matcher(index)
    rows: list[EvalRow] = []
    for item in labelled:
        scores: dict[str, float] = {}
        for name, scorer in scorers.items():
            results = search(index, item["query"], scorer, k, matcher)
            scores[name] = precision_at_k(results, item["relevant"], k)
        rows.append(EvalRow(item["query"], scores))
    return rows


def format_table(rows: Sequence[EvalRow], names: Sequence[str], k: int = 5) -> str:
    lines = [
        "| Запит | " + " | ".join(f"P@{k} {n}" for n in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for row in rows:
        lines.append(f"| {row.query} | " + " | ".join(f"{row.scores[n]:.2f}" for n in names) + " |")
    mean = {n: sum(r.scores[n] for r in rows) / len(rows) for n in names}
    lines.append("| **Середнє** | " + " | ".join(f"**{mean[n]:.2f}**" for n in names) + " |")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="precision@k: TF-IDF vs BM25")
    parser.add_argument("index_file")
    parser.add_argument("queries_file")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    fmt: IndexFormat = "json" if args.format == "json" else "pickle"
    k: int = args.k

    with open(args.queries_file, encoding="utf-8") as f:
        labelled = cast(list[LabelledQuery], json.load(f))

    scorers: dict[str, Scorer] = {"TF-IDF": TfIdf(), "BM25": BM25()}
    with open_index(args.index_file, fmt) as index:
        rows = evaluate(index, labelled, scorers, k)
    print(format_table(rows, list(scorers), k))


if __name__ == "__main__":
    main()
