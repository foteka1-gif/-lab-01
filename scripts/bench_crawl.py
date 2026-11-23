"""Бенчмарк асинхронного краулера для concurrency 1, 5 і 20 на 200+ сторінках."""

import argparse
import asyncio
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Protocol, cast

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from mock_site import MockSiteServer  # type: ignore[import-not-found]  # noqa: E402

from findex.crawler import crawl  # noqa: E402


class _MemInfo(Protocol):
    rss: int


class _ProcProto(Protocol):
    def memory_info(self) -> _MemInfo: ...


class _PsutilProto(Protocol):
    def Process(self, pid: int | None = None) -> _ProcProto: ...


def _peak_rss_mb() -> float:
    try:
        mod = cast(_PsutilProto, cast(object, importlib.import_module("psutil")))
        proc = mod.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024 * 1024)
    except Exception:
        return 0.0


async def _measure_once(
    base_url: str,
    *,
    max_pages: int,
    concurrency: int,
) -> tuple[float, float, int, float]:
    t0 = time.perf_counter()
    fetched = 0
    errors = 0

    async for page in crawl(
        base_url,
        max_pages=max_pages,
        concurrency=concurrency,
        per_host_concurrency=concurrency,
        min_delay_per_host=0.0,
        timeout_seconds=5.0,
    ):
        if page.error is None:
            fetched += 1
        else:
            errors += 1

    wall = time.perf_counter() - t0
    pps = fetched / max(1e-6, wall)
    rss = _peak_rss_mb()
    return wall, pps, errors, rss


async def main_async(max_pages: int, latency: float, levels: list[int]) -> None:
    site = MockSiteServer(total_pages=max_pages + 60, latency=latency)
    server = await asyncio.start_server(site.client_connected, "127.0.0.1", 0)
    first_sock = next(iter(server.sockets))
    _, port = cast(tuple[str, int], first_sock.getsockname())
    base_url = f"http://127.0.0.1:{port}/page/0"

    print(f"Запуск бенчмарку на {base_url} (max_pages={max_pages}, затримка={latency} с)...\n")
    rows: list[tuple[int, float, float, int, float]] = []

    try:
        for conc in levels:
            wall, pps, errors, rss = await _measure_once(
                base_url,
                max_pages=max_pages,
                concurrency=conc,
            )
            rows.append((conc, wall, pps, errors, rss))
    finally:
        server.close()

    print("| Concurrency | Wall, с | Сторінок/с | Помилки | Peak RSS, МБ |")
    print("| :---: | :---: | :---: | :---: | :---: |")
    for conc, wall, pps, errors, rss in rows:
        print(f"| {conc} | {wall:.2f} | {pps:.1f} | {errors} | {rss:.0f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк краулера findex.")
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--latency", type=float, default=0.04)
    parser.add_argument("--concurrency", type=int, nargs="+", default=(1, 5, 20))
    args = parser.parse_args()
    asyncio.run(main_async(args.max_pages, args.latency, list(args.concurrency)))


if __name__ == "__main__":
    main()
