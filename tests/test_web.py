"""Тести вебшару FastAPI через TestClient і dependency_overrides (без файлу індексу)."""

from typing import cast

from fastapi.testclient import TestClient

from findex import DocMeta, Posting
from findex.model import Index
from findex.search import Matcher
from findex.web.app import create_app, get_index, get_matcher
from findex.web.config import Settings


def _sample_index() -> Index:
    postings = {
        "python": [
            Posting(doc_id=0, tf=3, positions=(0, 4, 9)),
            Posting(doc_id=1, tf=1, positions=(2,)),
        ],
        "asyncio": [
            Posting(doc_id=0, tf=2, positions=(1, 5)),
        ],
        "fastapi": [
            Posting(doc_id=1, tf=2, positions=(0, 3)),
            Posting(doc_id=2, tf=1, positions=(1,)),
        ],
    }
    doc_lengths = {0: 12, 1: 8, 2: 5}
    doc_meta = {
        0: DocMeta(path="virtual://doc0.txt", title="Асинхронний Python"),
        1: DocMeta(path="virtual://doc1.txt", title="Веб на FastAPI"),
        2: DocMeta(path="virtual://doc2.txt", title="Мікросервіси"),
    }
    return Index(postings, doc_lengths, doc_meta)


def test_health_returns_503_when_index_not_loaded() -> None:
    app = create_app(Settings(index_path=None))
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 503
        data = cast(dict[str, object], resp.json())
        assert data["status"] == "unavailable"
        assert data["index_loaded"] is False


def test_search_pagination_stats_and_docs_with_overrides() -> None:
    app = create_app(Settings(index_path=None))
    fake_index = _sample_index()
    fake_matcher = Matcher(fake_index)

    app.dependency_overrides[get_index] = lambda: fake_index
    app.dependency_overrides[get_matcher] = lambda: fake_matcher

    with TestClient(app) as client:
        stats_resp = client.get("/stats")
        assert stats_resp.status_code == 200
        stats_data = cast(dict[str, object], stats_resp.json())
        assert stats_data["num_docs"] == 3

        s1 = client.get("/search", params={"q": "fastapi", "k": 1, "page": 1, "scorer": "bm25"})
        assert s1.status_code == 200
        body1 = cast(dict[str, object], s1.json())
        assert body1["total_hits"] == 2
        assert body1["total_pages"] == 2
        results1 = cast(list[dict[str, object]], body1["results"])
        assert len(results1) == 1
        first_hit = next(iter(results1))
        assert first_hit["doc_id"] == 1

        s2 = client.get("/search", params={"q": "fastapi", "k": 1, "page": 2})
        assert s2.status_code == 200
        body2 = cast(dict[str, object], s2.json())
        results2 = cast(list[dict[str, object]], body2["results"])
        second_hit = next(iter(results2))
        assert second_hit["doc_id"] == 2

        doc_ok = client.get("/docs/0")
        assert doc_ok.status_code == 200
        doc_data = cast(dict[str, object], doc_ok.json())
        assert doc_data["title"] == "Асинхронний Python"

        doc_missing = client.get("/docs/999")
        assert doc_missing.status_code == 404


def test_search_validation_422_on_bad_params_or_broken_query() -> None:
    app = create_app(Settings(index_path=None))
    fake_index = _sample_index()
    app.dependency_overrides[get_index] = lambda: fake_index
    app.dependency_overrides[get_matcher] = lambda: Matcher(fake_index)

    with TestClient(app) as client:
        r_empty = client.get("/search", params={"q": ""})
        assert r_empty.status_code == 422

        r_scorer = client.get("/search", params={"q": "python", "scorer": "unknown"})
        assert r_scorer.status_code == 422

        r_syntax = client.get("/search", params={"q": "python AND (asyncio"})
        assert r_syntax.status_code == 422


def test_html_home_page_renders_results_and_pagination() -> None:
    app = create_app(Settings(index_path=None))
    fake_index = _sample_index()

    with TestClient(app) as client:
        app.state.index = fake_index
        app.state.matcher = Matcher(fake_index)
        resp = client.get("/", params={"q": "fastapi", "k": 1, "page": 1})
        assert resp.status_code == 200
        assert "Веб на FastAPI" in resp.text
        assert "Наступна →" in resp.text


def test_unhandled_exception_returns_500_without_traceback() -> None:
    app = create_app(Settings(index_path=None))

    def broken_index() -> Index:
        raise RuntimeError("secret internal failure")

    app.dependency_overrides[get_index] = broken_index

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/stats")
        assert resp.status_code == 500
        data = cast(dict[str, object], resp.json())
        assert "request_id" in data
        assert "secret internal failure" not in resp.text
        assert "Traceback" not in resp.text
