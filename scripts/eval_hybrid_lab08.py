"""Оцінка precision@5 на 15 запитах (10 з лаби 3 + 5 перефразованих) для BM25, Semantic і Hybrid (RRF)."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from findex.index import build_index, make_pipeline  # noqa: E402
from findex.model import Index  # noqa: E402
from findex.search import search  # noqa: E402
from findex.semantic import (  # noqa: E402
    SemanticIndex,
    build_semantic_index,
    hybrid_search,
    semantic_search,
)


def _precision_at_5(predicted: list[int], relevant: set[int]) -> float:
    top5 = predicted[:5]
    if not top5:
        return 0.0
    hits = sum(1 for doc_id in top5 if doc_id in relevant)
    return hits / len(top5)


def main() -> None:
    idx: Index = build_index(make_pipeline(ROOT_DIR / "data"), positions=True)
    sem: SemanticIndex = build_semantic_index(idx, chunk_words=80, overlap=20)

    all_docs = set(idx.doc_ids)
    angel_docs = {
        d for d in idx.doc_ids if any(p.doc_id == d for p in idx.get("ангел", []))
    } or all_docs
    paradise_docs = {
        d for d in idx.doc_ids if any(p.doc_id == d for p in idx.get("рай", []))
    } or all_docs
    chaos_docs = {
        d for d in idx.doc_ids if any(p.doc_id == d for p in idx.get("хаос", []))
    } or all_docs
    light_docs = {
        d for d in idx.doc_ids if any(p.doc_id == d for p in idx.get("світло", []))
    } or all_docs

    eval_queries: list[tuple[str, str, set[int]]] = [
        ("Лаба 3 #1", "ангел", angel_docs),
        ("Лаба 3 #2", "рай", paradise_docs),
        ("Лаба 3 #3", "хаос", chaos_docs),
        ("Лаба 3 #4", "світло", light_docs),
        ("Лаба 3 #5", "ангел AND рай", (angel_docs & paradise_docs) or angel_docs),
        ("Лаба 3 #6", "ангел OR хаос", angel_docs | chaos_docs),
        ("Лаба 3 #7", "світло OR рай", light_docs | paradise_docs),
        ("Лаба 3 #8", "небо", all_docs),
        ("Лаба 3 #9", "люцифер", chaos_docs),
        ("Лаба 3 #10", "ворота", paradise_docs),
        ("Перефразований #11", "небесні посланці та крилаті охоронці", angel_docs),
        ("Перефразований #12", "вхід до блаженної оселі праведників", paradise_docs),
        ("Перефразований #13", "первісна безодня та падіння бунтівного духа", chaos_docs),
        ("Перефразований #14", "сяйво творіння у темряві всесвіту", light_docs),
        (
            "Перефразований #15",
            "протистояння небесного воїнства і темних сил",
            angel_docs | chaos_docs,
        ),
    ]

    bm25_scores: list[float] = []
    sem_scores: list[float] = []
    hyb_scores: list[float] = []

    print("| Група | Запит | BM25 P@5 | Semantic P@5 | Hybrid (RRF) P@5 |")
    print("| :--- | :--- | :---: | :---: | :---: |")

    for group, q, rel in eval_queries:
        try:
            lex_hits = search(idx, q, k=5)
            lex_pool = search(idx, q, k=15)
        except ValueError:
            lex_hits = []
            lex_pool = []
        sem_hits = semantic_search(idx, sem, q, k=5)
        hyb_hits = hybrid_search(idx, sem, lex_pool, q, k=5)

        p_bm25 = _precision_at_5([r.doc_id for r in lex_hits], rel)
        p_sem = _precision_at_5([r.doc_id for r in sem_hits], rel)
        p_hyb = _precision_at_5([r.doc_id for r in hyb_hits], rel)

        bm25_scores.append(p_bm25)
        sem_scores.append(p_sem)
        hyb_scores.append(p_hyb)
        print(f"| {group} | `{q}` | {p_bm25:.2f} | {p_sem:.2f} | {p_hyb:.2f} |")

    print(
        f"| **Середнє (усі 15 запитів)** | — | **{sum(bm25_scores) / len(bm25_scores):.3f}** | "
        f"**{sum(sem_scores) / len(sem_scores):.3f}** | **{sum(hyb_scores) / len(hyb_scores):.3f}** |"
    )
    print(
        f"| **Середнє (5 перефразованих)** | — | **{sum(bm25_scores[10:]) / 5:.3f}** | "
        f"**{sum(sem_scores[10:]) / 5:.3f}** | **{sum(hyb_scores[10:]) / 5:.3f}** |"
    )


if __name__ == "__main__":
    main()
