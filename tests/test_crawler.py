"""Тести для асинхронного краулера (без живої мережі, через httpx.MockTransport)."""

import asyncio
import warnings
from pathlib import Path

import httpx

from findex.corpus import iter_documents, write_jsonl_record
from findex.crawler import CrawledPage, crawl
from findex.fetch import fetch_url
from findex.htmlparse import parse_html
from findex.index import build_index, make_pipeline
from findex.robots import RobotsCache
from findex.search import search
from findex.urls import canonicalize_url


def test_canonicalize_url_normalizes_scheme_host_fragment_slash_and_query() -> None:
    from urllib.parse import urlunsplit

    raw_1 = urlunsplit(("HTTP", "Example.COM:80", "/docs/page/../intro/", "b=2&a=1", "sec1"))
    raw_2 = urlunsplit(("http", "example.com", "/docs/intro/", "a=1&b=2", "sec2"))
    expected = urlunsplit(("http", "example.com", "/docs/intro/", "a=1&b=2", ""))

    u1 = canonicalize_url(raw_1)
    u2 = canonicalize_url(raw_2)
    assert u1 == expected
    assert u1 == u2


    assert canonicalize_url("http" + "://example.com") == "http" + "://example.com/"
    assert canonicalize_url("mailto:test@example.com") is None
    assert canonicalize_url("javascript:void(0)") is None

    base_site = "https" + "://site.org/dir/index.html"
    rel = canonicalize_url("/sub/item", base_url=base_site)
    assert rel == "https" + "://site.org/sub/item"


def test_parse_html_extracts_title_visible_text_and_links() -> None:
    ext_link = "https" + "://example.org/about#team"
    html = f"""
    <html>
      <head>
        <title>  Тестова   сторінка </title>
        <style>.hidden {{ display: none; }}</style>
        <script>console.log("secret_js_token");</script>
      </head>
      <body>
        <h1>Привіт, світ!</h1>
        <p>Асинхронний пошук у Python.</p>
        <a href="/next">Далі</a>
        <a href="{ext_link}">Про нас</a>
      </body>
    </html>
    """
    parsed = parse_html(html)
    assert parsed.title == "Тестова сторінка"
    assert "Привіт, світ!" in parsed.text
    assert "Асинхронний пошук у Python." in parsed.text
    assert "secret_js_token" not in parsed.text
    assert parsed.links == ("/next", ext_link)


def test_fetch_retries_429_and_5xx_but_not_404() -> None:
    async def run() -> None:
        calls: dict[str, int] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            calls[path] = calls.get(path, 0) + 1
            if path == "/rate":
                if calls[path] == 1:
                    return httpx.Response(429, headers={"Retry-After": "0.01"}, text="Slow down")
                return httpx.Response(200, headers={"Content-Type": "text/html"}, text="OK 429")
            if path == "/err503":
                if calls[path] <= 2:
                    return httpx.Response(503, text="Temp fail")
                return httpx.Response(200, headers={"Content-Type": "text/html"}, text="OK 503")
            if path == "/missing":
                return httpx.Response(404, text="Not found")
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        base = "https" + "://test.local"
        async with httpx.AsyncClient(transport=transport) as client:
            r429 = await fetch_url(client, base + "/rate", base_backoff=0.01)
            assert r429.ok
            assert r429.attempts == 2
            assert r429.text == "OK 429"

            r503 = await fetch_url(client, base + "/err503", base_backoff=0.01)
            assert r503.ok
            assert r503.attempts == 3

            r404 = await fetch_url(client, base + "/missing", base_backoff=0.01)
            assert not r404.ok
            assert r404.status == 404
            assert r404.attempts == 1
            assert calls["/missing"] == 1

    asyncio.run(run())


def test_robots_cache_per_host_and_fetches_once() -> None:
    async def run() -> None:
        robots_hits = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal robots_hits
            if request.url.path == "/robots.txt":
                robots_hits += 1
                return httpx.Response(200, text="User-agent: *\nDisallow: /secret/\n")
            return httpx.Response(200, text="ok")

        transport = httpx.MockTransport(handler)
        base = "https" + "://docs.local"
        async with httpx.AsyncClient(transport=transport) as client:
            cache = RobotsCache(client)
            assert await cache.can_fetch(base + "/guide") is True
            assert await cache.can_fetch(base + "/secret/key") is False
            assert await cache.can_fetch(base + "/other") is True
            assert robots_hits == 1

    asyncio.run(run())


def test_crawl_respects_max_pages_dedup_robots_and_no_blocking() -> None:
    async def run() -> None:
        ext_url = "https" + "://external.org/out"

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nDisallow: /blocked\n")
            if path == "/blocked":
                return httpx.Response(200, headers={"Content-Type": "text/html"}, text="Blocked")
            idx = 0 if path == "/" else int(path.removeprefix("/p/"))
            html = (
                f"<html><head><title>Page {idx}</title></head><body>"
                f"Document {idx} content."
                f'<a href="/p/{idx + 1}#frag">next</a>'
                f'<a href="/p/{idx + 1}?x=1&y=2">dup_or_q</a>'
                f'<a href="/blocked">blocked</a>'
                f'<a href="{ext_url}">external</a>'
                "</body></html>"
            )
            return httpx.Response(200, headers={"Content-Type": "text/html"}, text=html)

        transport = httpx.MockTransport(handler)
        pages: list[CrawledPage] = []
        seed = "https" + "://mock.local/"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            async for page in crawl(
                seed,
                max_pages=6,
                concurrency=3,
                per_host_concurrency=3,
                min_delay_per_host=0.0,
                transport=transport,
            ):
                if page.error is None:
                    pages.append(page)

        assert len(pages) == 6
        assert all("/blocked" not in p.url for p in pages)
        assert all("external.org" not in p.url for p in pages)
        slow_callbacks = [w for w in caught if "executing" in str(w.message).lower()]
        assert not slow_callbacks

    asyncio.run(run(), debug=True)


def test_jsonl_roundtrip_into_index_and_search(tmp_path: Path) -> None:
    jsonl_file = tmp_path / "crawled.jsonl"
    with jsonl_file.open("w", encoding="utf-8") as f:
        write_jsonl_record(
            f,
            url="https" + "://mock.local/a",
            title="Асинхронний Python",
            text="Цикл подій asyncio та корутини працюють в одному потоці.",
        )
        write_jsonl_record(
            f,
            url="https" + "://mock.local/b",
            title="Пошуковий рушій",
            text="Інвертований індекс підтримує булевий пошук та фразу.",
        )

    docs = list(iter_documents(jsonl_file))
    assert len(docs) == 2
    first_doc = next(iter(docs))
    assert first_doc.title == "Асинхронний Python"

    idx = build_index(make_pipeline(jsonl_file), positions=True)
    results = search(idx, "корутини")
    assert len(results) == 1
    first_hit = next(iter(results))
    assert first_hit.title == "Асинхронний Python"
