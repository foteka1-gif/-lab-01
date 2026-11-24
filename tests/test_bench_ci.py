"""Тести NumPy-векторизації, семантичного/гібридного RRF-пошуку та pytest-benchmark для CI."""

from pathlib import Path
from typing import Any

from findex import DocMeta, Posting
from findex.index import build_index, make_pipeline
from findex.model import Index
from findex.scoring import BM25, TfIdf
from findex.search import search, search_legacy
from findex.semantic import build_semantic_index, hybrid_search, semantic_search


def _sample_index() -> Index:
    postings = {
        "ангел": [
            Posting(doc_id=0, tf=5, positions=(0, 3, 7, 11, 15)),
            Posting(doc_id=1, tf=2, positions=(1, 8)),
            Posting(doc_id=2, tf=1, positions=(4,)),
        ],
        "небо": [
            Posting(doc_id=0, tf=2, positions=(2, 9)),
            Posting(doc_id=2, tf=3, positions=(0, 5, 10)),
        ],
        "рай": [
            Posting(doc_id=1, tf=4, positions=(0, 3, 6, 9)),
        ],
    }
    doc_lengths = {0: 25, 1: 18, 2: 20}
    doc_meta = {
        0: DocMeta(path="virtual://0.txt", title="Світло небес"),
        1: DocMeta(path="virtual://1.txt", title="Ворота в рай"),
        2: DocMeta(path="virtual://2.txt", title="Хроніки вічності"),
    }
    return Index(postings, doc_lengths, doc_meta)


def test_numpy_search_matches_legacy_order_and_scores() -> None:
    idx = _sample_index()
    for scorer in (BM25(), TfIdf()):
        for query in ("ангел", "ангел OR небо OR рай", "ангел AND небо"):
            legacy = search_legacy(idx, query, scorer=scorer, k=10)
            vectorized = search(idx, query, scorer=scorer, k=10)
            assert [r.doc_id for r in vectorized] == [r.doc_id for r in legacy]
            for r_vec, r_leg in zip(vectorized, legacy, strict=True):
                assert abs(r_vec.score - r_leg.score) < 1e-9


def test_semantic_and_hybrid_rrf_search() -> None:
    idx = _sample_index()
    sem = build_semantic_index(idx, chunk_words=20, overlap=5)
    sem_hits = semantic_search(idx, sem, "ворота раю", k=3)
    assert sem_hits
    assert next(iter(sem_hits)).doc_id == 1

    lex_hits = search(idx, "ангел OR рай", k=3)
    hyb_hits = hybrid_search(idx, sem, lex_hits, "ворота раю", k=3)
    assert hyb_hits
    assert {h.doc_id for h in hyb_hits}.issuperset({0, 1})


def test_benchmark_search_ci(benchmark: Any) -> None:
    idx = _sample_index()
    idx.warm_numpy_cache()
    res = benchmark(search, idx, "ангел OR небо OR рай", None, 10, None)
    assert len(res) == 3


def test_benchmark_build_index_ci(benchmark: Any, tmp_path: Path) -> None:
    sample = tmp_path / "doc.txt"
    sample.write_text("ангел світло небо рай " * 40, encoding="utf-8")
    idx = benchmark(lambda: build_index(make_pipeline(tmp_path), positions=True))
    assert idx.num_docs == 1
