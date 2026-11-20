"""Бенчмарк побудови індексу: виконавець × число воркерів.

Кожен замір іде в ОКРЕМОМУ процесі: піковий RSS — це максимум за весь час життя процесу,
тож у спільному процесі кожен наступний замір успадкував би пік попередніх.

    python scripts/bench_index.py data_big --repeats 3 --out results.json

На Python 3.13t (без GIL) запусти те саме тим інтерпретатором і збережи в інший файл:
    uv run --python 3.13t python scripts/bench_index.py data_big --out results_313t.json
"""
# pyright: basic

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from findex.corpus import list_document_paths
from findex.parallel import build_index_parallel


def peak_rss_bytes() -> int:
    """Піковий розмір резидентної пам'яті ЦЬОГО процесу (0, якщо дізнатись не вдалось)."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)  # є лише на Windows
        if peak is not None:
            return int(peak)
    except ImportError:
        pass
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore

        return usage if sys.platform == "darwin" else usage * 1024  # Linux: КБ, macOS: байти
    except ImportError:
        return 0


def run_single(corpus: Path, executor: str, workers: int, positions: bool) -> dict[str, Any]:
    paths = list_document_paths(corpus)
    report = build_index_parallel(
        paths,
        workers=workers,
        executor=executor,  # type: ignore[arg-type]  # рядок з argparse, перевірений choices
        positions=positions,
    )
    return {
        "wall": report.wall_seconds,
        "cpu": report.cpu_seconds,
        "merge": report.merge_seconds,
        "rss": peak_rss_bytes(),
        "docs": len(paths),
    }


def measure_in_child(corpus: Path, executor: str, workers: int, positions: bool) -> dict[str, Any]:
    command = [sys.executable, __file__, "--single", str(corpus), executor, str(workers)]
    if positions:
        command.append("--positions")
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    if done.returncode != 0:
        raise RuntimeError(f"Замір {executor}/{workers} упав:\n{done.stderr}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def machine_info() -> dict[str, Any]:
    gil_check = getattr(sys, "_is_gil_enabled", None)  # з'явилось в Python 3.13
    return {
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gil_enabled": gil_check() if gil_check is not None else True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк findex index.")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=None)
    parser.add_argument(
        "--executors", nargs="+", default=["serial", "threads", "processes"],
        choices=["serial", "threads", "processes"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--positions", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("executor_arg", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("workers_arg", nargs="?", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.single:  # режим «дитини»: один замір, JSON в stdout
        result = run_single(args.corpus, args.executor_arg, args.workers_arg, args.positions)
        print(json.dumps(result))
        return

    cpu_count = os.cpu_count() or 1
    worker_counts = sorted(set(args.workers or [1, 2, 4, 8, cpu_count]))

    configs: list[tuple[str, int]] = []
    for executor in args.executors:
        if executor == "serial":
            configs.append(("serial", 1))  # послідовний варіант не має воркерів
        else:
            configs.extend((executor, n) for n in worker_counts)

    print("Розігрів (наповнюємо кеш файлової системи)...", file=sys.stderr)
    measure_in_child(args.corpus, "serial", 1, args.positions)

    rows: list[dict[str, Any]] = []
    for executor, workers in configs:
        runs = []
        for run_number in range(args.repeats):
            print(f"{executor:9} workers={workers:<3} run {run_number + 1}/{args.repeats}",
                  file=sys.stderr)
            runs.append(measure_in_child(args.corpus, executor, workers, args.positions))
        rows.append({
            "executor": executor,
            "workers": workers,
            "wall": statistics.median(r["wall"] for r in runs),
            "cpu": statistics.median(r["cpu"] for r in runs),
            "merge": statistics.median(r["merge"] for r in runs),
            "rss_mb": statistics.median(r["rss"] for r in runs) / 1024 / 1024,
            "runs": runs,
        })

    baseline = next((r["wall"] for r in rows if r["executor"] == "serial"), rows[0]["wall"])
    for row in rows:
        row["speedup"] = baseline / row["wall"]

    result = {
        "machine": machine_info(),
        "corpus": {"path": str(args.corpus), "docs": rows[0]["runs"][0]["docs"]},
        "positions": args.positions,
        "repeats": args.repeats,
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Збережено: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
