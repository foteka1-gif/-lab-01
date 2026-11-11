"""M4: precision@5 для TF-IDF vs BM25 на ваших 10 розмічених запитах.
 
eval_queries.json:
  [{"query": "python AND async", "relevant": ["назва_файлу_без_txt", "інша"]}, ...]
"""
import argparse
import json
 
from .model import open_index
from .scoring import BM25, TfIdf
from .search import search, make_matcher
 
 
def precision_at_k(results, relevant_titles, k=5):
    """Скільки з топ-k результатів розмічено як правильні, поділити на k."""
    relevant = set(relevant_titles)
    return sum(1 for r in results[:k] if r.title in relevant) / k
 
 
def evaluate(index, labelled, scorers, k=5):
    matcher = make_matcher(index)
    rows = []
    for item in labelled:
        row = {"query": item["query"]}
        for name, scorer in scorers.items():
            results = search(index, item["query"], scorer, k, matcher)
            row[name] = precision_at_k(results, item["relevant"], k)
        rows.append(row)
    return rows
 
 
def format_table(rows, names, k=5):
    lines = [f"| Запит | " + " | ".join(f"P@{k} {n}" for n in names) + " |",
             "|---|" + "---|" * len(names)]
    for row in rows:
        lines.append(f"| {row['query']} | " + " | ".join(f"{row[n]:.2f}" for n in names) + " |")
    mean = {n: sum(r[n] for r in rows) / len(rows) for n in names}
    lines.append("| **Середнє** | " + " | ".join(f"**{mean[n]:.2f}**" for n in names) + " |")
    return "\n".join(lines)
 
 
def main():
    parser = argparse.ArgumentParser(description="precision@k: TF-IDF vs BM25")
    parser.add_argument("index_file")
    parser.add_argument("queries_file")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
 
    with open(args.queries_file, encoding="utf-8") as f:
        labelled = json.load(f)
 
    scorers = {"TF-IDF": TfIdf(), "BM25": BM25()}
    with open_index(args.index_file, args.format) as index:
        rows = evaluate(index, labelled, scorers, args.k)
    print(format_table(rows, list(scorers), args.k))
 
 
if __name__ == "__main__":
    main()