"""Асинхронний веб-краулер на asyncio, TaskGroup, чергах та семафорах."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from .fetch import DEFAULT_USER_AGENT, fetch_url
from .htmlparse import parse_html
from .robots import RobotsCache
from .urls import canonicalize_url, same_host

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CrawlStats:
    """Поточний стан краулера для живих лічильників CLI."""

    fetched: int = 0
    errors: int = 0
    skipped: int = 0
    in_flight: int = 0
    queued: int = 0
    bytes_downloaded: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return max(1e-6, time.perf_counter() - self.started_at)

    @property
    def pages_per_second(self) -> float:
        return self.fetched / self.elapsed


@dataclass(frozen=True, slots=True)
class CrawledPage:
    """Результат обробки одного URL (успішна сторінка або запис про помилку/пропуск)."""

    url: str
    final_url: str
    status: int
    title: str
    text: str
    num_bytes: int
    elapsed: float
    attempts: int
    error: str | None
    stats: CrawlStats


class _HostLimiter:
    """Обмежує кількість одночасних запитів і мінімальну паузу між запитами до одного хоста."""

    def __init__(self, per_host_limit: int, min_delay: float) -> None:
        self._per_host_limit = max(1, per_host_limit)
        self._min_delay = max(0.0, min_delay)
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._rate_locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}

    def semaphore_for(self, host: str) -> asyncio.Semaphore:
        sem = self._semaphores.get(host)
        if sem is None:
            sem = asyncio.Semaphore(self._per_host_limit)
            self._semaphores[host] = sem
        return sem

    async def wait_turn(self, host: str) -> None:
        if self._min_delay <= 0.0:
            return
        lock = self._rate_locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.perf_counter()
            last = self._last_request.get(host)
            if last is not None:
                remaining = self._min_delay - (now - last)
                if remaining > 0.0:
                    await asyncio.sleep(remaining)
            self._last_request[host] = time.perf_counter()


def _host_key(url: str) -> str:
    parts = urlsplit(url)
    return parts.netloc.lower()


async def crawl(
    start_url: str,
    *,
    max_pages: int = 50,
    concurrency: int = 5,
    per_host_concurrency: int = 2,
    min_delay_per_host: float = 0.1,
    timeout_seconds: float = 10.0,
    max_retries: int = 3,
    same_domain_only: bool = True,
    respect_robots: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[CrawledPage]:
    """Асинхронний генератор обходу сайту з лімітом max_pages та паралельними воркерами."""
    seed = canonicalize_url(start_url)
    if seed is None:
        raise ValueError(f"Некоректний початковий URL: {start_url}")
    if max_pages < 1:
        raise ValueError("max_pages має бути >= 1")
    if concurrency < 1:
        raise ValueError("concurrency має бути >= 1")

    stats = CrawlStats()
    seen: set[str] = {seed}
    url_queue: asyncio.Queue[str | None] = asyncio.Queue()
    results_queue: asyncio.Queue[CrawledPage | None] = asyncio.Queue()

    url_queue.put_nowait(seed)
    stats.queued = 1

    global_sem = asyncio.Semaphore(concurrency)
    host_limiter = _HostLimiter(per_host_concurrency, min_delay_per_host)
    stop_event = asyncio.Event()
    reserved_pages = 0

    async def worker(client: httpx.AsyncClient, robots: RobotsCache) -> None:
        nonlocal reserved_pages
        while not stop_event.is_set():
            url = await url_queue.get()
            if url is None:
                url_queue.task_done()
                return

            stats.queued = url_queue.qsize()
            if stop_event.is_set():
                url_queue.task_done()
                return

            if respect_robots and not await robots.can_fetch(url):
                stats.skipped += 1
                await results_queue.put(
                    CrawledPage(
                        url=url,
                        final_url=url,
                        status=0,
                        title="",
                        text="",
                        num_bytes=0,
                        elapsed=0.0,
                        attempts=0,
                        error="disallowed_by_robots",
                        stats=stats,
                    )
                )
                url_queue.task_done()
                continue

            if reserved_pages >= max_pages:
                stop_event.set()
                url_queue.task_done()
                return
            reserved_pages += 1

            host = _host_key(url)
            host_sem = host_limiter.semaphore_for(host)

            async with global_sem, host_sem:
                if stop_event.is_set():
                    url_queue.task_done()
                    return
                await host_limiter.wait_turn(host)
                stats.in_flight += 1
                try:
                    fetched = await fetch_url(
                        client,
                        url,
                        timeout_seconds=timeout_seconds,
                        max_retries=max_retries,
                    )
                finally:
                    stats.in_flight -= 1

            if fetched.ok:
                stats.fetched += 1
                stats.bytes_downloaded += fetched.num_bytes
                is_html = "html" in fetched.content_type.lower() or not fetched.content_type
                parsed = parse_html(fetched.text) if is_html else None
                title = parsed.title if parsed else ""
                text = parsed.text if parsed else fetched.text

                if parsed and not stop_event.is_set():
                    for raw_link in parsed.links:
                        cand = canonicalize_url(raw_link, base_url=fetched.final_url)
                        if cand is None or cand in seen:
                            continue
                        if same_domain_only and not same_host(seed, cand):
                            continue
                        seen.add(cand)
                        url_queue.put_nowait(cand)
                stats.queued = url_queue.qsize()

                await results_queue.put(
                    CrawledPage(
                        url=url,
                        final_url=fetched.final_url,
                        status=fetched.status,
                        title=title,
                        text=text,
                        num_bytes=fetched.num_bytes,
                        elapsed=fetched.elapsed,
                        attempts=fetched.attempts,
                        error=None,
                        stats=stats,
                    )
                )
            else:
                stats.errors += 1
                reserved_pages = max(0, reserved_pages - 1)
                stats.queued = url_queue.qsize()
                await results_queue.put(
                    CrawledPage(
                        url=url,
                        final_url=fetched.final_url,
                        status=fetched.status,
                        title="",
                        text="",
                        num_bytes=fetched.num_bytes,
                        elapsed=fetched.elapsed,
                        attempts=fetched.attempts,
                        error=fetched.error or f"HTTP {fetched.status}",
                        stats=stats,
                    )
                )

            url_queue.task_done()
            if stats.fetched >= max_pages:
                stop_event.set()

    async def coordinator() -> None:
        async with httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            transport=transport,
        ) as client:
            robots = RobotsCache(client, user_agent=user_agent, timeout_seconds=timeout_seconds)
            try:
                async with asyncio.TaskGroup() as tg:
                    worker_tasks = [
                        tg.create_task(worker(client, robots)) for _ in range(concurrency)
                    ]
                    join_task = asyncio.create_task(url_queue.join())
                    stop_task = asyncio.create_task(stop_event.wait())

                    done, pending = await asyncio.wait(
                        {join_task, stop_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        _ = task.exception()

                    stop_event.set()
                    for task in worker_tasks:
                        task.cancel()
            except* asyncio.CancelledError:
                pass
            finally:
                await results_queue.put(None)

    coord_task = asyncio.create_task(coordinator())
    try:
        while True:
            item = await results_queue.get()
            if item is None:
                break
            yield item
            if item.stats.fetched >= max_pages:
                stop_event.set()

    finally:
        stop_event.set()
        if not coord_task.done():
            coord_task.cancel()
        await asyncio.gather(coord_task, return_exceptions=True)
