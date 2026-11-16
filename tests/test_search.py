import pytest

from findex.model import Index
from findex.scoring import BM25, TfIdf
from findex.search import Matcher, SearchResult, search
from findex.snippets import make_snippet


def titles(results: list[SearchResult]) -> list[str]:
    return [r.title for r in results]


def test_search_orders_by_score_then_doc_id(index: Index) -> None:
    # async: tf=1 в обох, довжини однакові -> рівні бали -> менший doc_id першим
    assert titles(search(index, "async", BM25())) == ["a_alpha", "c_gamma"]


def test_search_limits_k(index: Index) -> None:
    assert len(search(index, "python", BM25(), k=2)) == 2


def test_search_no_match(index: Index) -> None:
    assert search(index, "zzz") == []


def test_search_not_query_has_zero_scores(index: Index) -> None:
    results = search(index, "NOT python")
    assert titles(results) == ["e_epsilon"]
    assert results[0].score == 0.0


def test_search_phrase(index: Index) -> None:
    assert titles(search(index, '"async await"')) == ["a_alpha"]


def test_search_uses_default_scorer(index: Index) -> None:
    assert search(index, "rust") == search(index, "rust", BM25())


@pytest.mark.parametrize("scorer", [BM25(), TfIdf()], ids=repr)
def test_best_first(index: Index, scorer: BM25 | TfIdf) -> None:
    scores = [r.score for r in search(index, "python", scorer)]
    assert scores == sorted(scores, reverse=True)


def test_search_result_sorts_by_score() -> None:
    low, high = SearchResult(1.0, 0, "a"), SearchResult(2.0, 1, "b")
    assert max([low, high]) == high


def test_matcher_cache(index: Index) -> None:
    matcher = Matcher(index)
    assert matcher("python") == (0, 1, 2, 3)
    matcher("python")
    info = matcher.cache_info()
    assert (info.hits, info.misses) == (1, 1)
    matcher.cache_clear()
    assert matcher.cache_info().currsize == 0


def test_snippet_highlights_best_window() -> None:
    text = "x " * 100 + "python rocks " + "y " * 100
    snippet = make_snippet(text, ["python"], window=10)
    assert "**python**" in snippet
    assert snippet.startswith("…") and snippet.endswith("…")


def test_snippet_without_match_is_plain_head() -> None:
    assert make_snippet("hello   world", ["zzz"]) == "hello world"
