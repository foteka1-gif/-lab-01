from pathlib import Path

import pytest

from findex.benchmark import benchmark_search, measure_memory_all
from findex.evaluate import LabelledQuery, evaluate, format_table, precision_at_k
from findex.model import Index
from findex.pipeline import run_eager_pipeline, run_lazy_pipeline
from findex.scoring import BM25
from findex.search import SearchResult


def test_lazy_equals_eager(corpus_dir: Path) -> None:
    assert run_lazy_pipeline(corpus_dir) == run_eager_pipeline(corpus_dir)


def test_lazy_equals_eager_with_limit(corpus_dir: Path) -> None:
    assert run_lazy_pipeline(corpus_dir, 2) == run_eager_pipeline(corpus_dir, 2)
    assert sum(run_lazy_pipeline(corpus_dir, 1).values()) == 4


def test_precision_at_k() -> None:
    results = [SearchResult(1.0, 0, "a"), SearchResult(0.5, 1, "b")]
    assert precision_at_k(results, ["a"], k=5) == pytest.approx(0.2)


def test_evaluate_and_table(index: Index) -> None:
    labelled: list[LabelledQuery] = [{"query": "rust", "relevant": ["b_beta"]}]
    rows = evaluate(index, labelled, {"BM25": BM25()})
    assert rows[0].scores["BM25"] == pytest.approx(0.2)
    table = format_table(rows, ["BM25"])
    assert "**Середнє**" in table
    assert "0.20" in table


def test_memory_report(index: Index) -> None:
    report = measure_memory_all(index)
    assert report.postings == sum(len(v) for v in index.values())
    assert report.array_mb <= report.normal_mb


def test_benchmark_search_rows(index: Index) -> None:
    rows = benchmark_search(index, repeats=3)
    assert [r.label for r in rows] == ["Найрідкісніші", "Найчастіші"]
