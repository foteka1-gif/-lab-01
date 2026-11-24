"""Навантажувальний тест вебсервісу findex: def vs async def та перцентилі p50/p95/p99."""

import argparse
import asyncio
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import httpx

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from findex.index import build_index, make_pipeline  # noqa: E402
from findex.model import save_index  # noqa: E402


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round((pct / 100.0) * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


async def _burst_20(base_url: str) -> tuple[float, float]:
    """20 одночасних запитів /search + фоновий /health для перевірки блокування event loop."""
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        t0 = time.perf_counter()
        search_tasks = [
            client.get("/search", params={"q": "ангел OR люцифер OR рай", "k": 10})
            for _ in range(20)
        ]
        health_t0 = time.perf_counter()
        health_task = client.get("/health")
        results = await asyncio.gather(health_task, *search_tasks)
        health_ms = (time.perf_counter() - health_t0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0
        assert all(r.status_code == 200 for r in results)
        return total_ms, health_ms


async def _load_test(
    base_url: str, total_requests: int, concurrency: int
) -> tuple[float, float, float, float]:
    """Навантажувальний тест: повертає (RPS, p50_ms, p95_ms, p99_ms)."""
    sem = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        # Прогрів
        await client.get("/search", params={"q": "ангел", "k": 10})

        async def one_req(i: int) -> None:
            q = "ангел" if i % 2 == 0 else "люцифер OR рай"
            async with sem:
                t0 = time.perf_counter()
                resp = await client.get("/search", params={"q": q, "k": 10})
                dt = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    latencies_ms.append(dt)

        started = time.perf_counter()
        await asyncio.gather(*(one_req(i) for i in range(total_requests)))
        wall = max(1e-6, time.perf_counter() - started)

    latencies_ms.sort()
    rps = len(latencies_ms) / wall
    return (
        rps,
        _percentile(latencies_ms, 50),
        _percentile(latencies_ms, 95),
        _percentile(latencies_ms, 99),
    )


async def _wait_ready(base_url: str, timeout_s: float = 15.0) -> None:
    deadline = time.perf_counter() + timeout_s
    async with httpx.AsyncClient(base_url=base_url, timeout=2.0) as client:
        while time.perf_counter() < deadline:
            try:
                r = await client.get("/health")
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.2)
    raise RuntimeError(f"Сервер {base_url} не відповів /health=200 за {timeout_s} с")


def _ensure_index(index_path: Path) -> None:
    if index_path.is_file():
        return
    index_path.parent.mkdir(parents=True, exist_ok=True)
    idx = build_index(make_pipeline(ROOT_DIR / "data"), positions=True)
    save_index(idx, index_path, "pickle")


async def run_local_suite(total_requests: int, concurrency: int) -> None:
    index_path = ROOT_DIR / "data" / "index.bin"
    _ensure_index(index_path)

    # Тимчасовий обгортковий модуль для порівняння def vs async def
    wrapper_path = ROOT_DIR / "scripts" / "_bench_asgi_modes.py"
    wrapper_path.write_text(
        "from typing import Annotated\n"
        "from fastapi import Depends, Query\n"
        "from findex.model import Index\n"
        "from findex.search import Matcher\n"
        "from findex.web.app import _execute_search, create_app, get_index, get_matcher\n"
        "from findex.web.schemas import SearchQueryParams, SearchResponse\n"
        "\n"
        "def make_sync_app():\n"
        "    return create_app()\n"
        "\n"
        "def make_async_app():\n"
        "    app = create_app()\n"
        "    app.router.routes = [r for r in app.router.routes if getattr(r, 'path', '') != '/search']\n"
        "    @app.get('/search', response_model=SearchResponse)\n"
        "    async def async_search(\n"
        "        params: Annotated[SearchQueryParams, Query()],\n"
        "        index: Annotated[Index, Depends(get_index)],\n"
        "        matcher: Annotated[Matcher, Depends(get_matcher)],\n"
        "    ) -> SearchResponse:\n"
        "        return _execute_search(index, matcher, params)\n"
        "    return app\n",
        encoding="utf-8",
    )

    configs = [
        ("1 воркер (async def)", "scripts._bench_asgi_modes:make_async_app", 1, 8011),
        ("1 воркер (def)", "scripts._bench_asgi_modes:make_sync_app", 1, 8012),
        ("4 воркери (def)", "scripts._bench_asgi_modes:make_sync_app", 4, 8013),
    ]

    rows: list[tuple[str, float, float, float, float]] = []
    burst_rows: list[tuple[str, float, float]] = []

    try:
        for label, target, workers, port in configs:
            env = os.environ.copy()
            env["INDEX_PATH"] = str(index_path)
            env["PYTHONPATH"] = f"{ROOT_DIR / 'src'}{os.pathsep}{ROOT_DIR}"
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    target,
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--workers",
                    str(workers),
                    "--log-level",
                    "error",
                ],
                env=env,
            )
            base_url = f"http://127.0.0.1:{port}"
            try:
                await _wait_ready(base_url)
                if workers == 1:
                    b_total, b_health = await _burst_20(base_url)
                    burst_rows.append((label, b_total, b_health))
                rps, p50, p95, p99 = await _load_test(base_url, total_requests, concurrency)
                rows.append((label, rps, p50, p95, p99))
            finally:
                proc.terminate()
                proc.wait(timeout=10)
    finally:
        if wrapper_path.exists():
            wrapper_path.unlink()

    print("\n### 20 одночасних запитів /search + паралельний /health:")
    for label, b_total, b_health in burst_rows:
        print(f"- {label}: 20 запитів за {b_total:.1f} мс, затримка /health = {b_health:.1f} мс")

    print("\n### Таблиця навантажувального тесту:")
    print("| Конфігурація | Запитів/с (RPS) | p50, мс | p95, мс | p99, мс |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for label, rps, p50, p95, p99 in rows:
        print(f"| {label} | {rps:.1f} | {p50:.2f} | {p95:.2f} | {p99:.2f} |")
    _ = statistics.mean([1.0])


async def run_remote(url: str, total_requests: int, concurrency: int) -> None:
    clean_url = url.rstrip("/")
    await _wait_ready(clean_url, timeout_s=30.0)
    rps, p50, p95, p99 = await _load_test(clean_url, total_requests, concurrency)
    print(f"| Розгорнутий сервіс ({clean_url}) | {rps:.1f} | {p50:.2f} | {p95:.2f} | {p99:.2f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Навантажувальний тест вебсервісу findex.")
    parser.add_argument("--url", default="", help="Публічний URL розгорнутого сервісу (опційно).")
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    url_val = cast(str, args.url).strip()
    reqs = cast(int, args.requests)
    conc = cast(int, args.concurrency)

    if url_val:
        asyncio.run(run_remote(url_val, reqs, conc))
    else:
        asyncio.run(run_local_suite(reqs, conc))


if __name__ == "__main__":
    main()
