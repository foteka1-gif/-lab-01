"""Порівняння швидкості скорера лаби 3 та NumPy (int32 + argpartition)."""

import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from findex.index import build_index, make_pipeline  # noqa: E402
from findex.model import Index  # noqa: E402
from findex.search import SearchResult, search, search_legacy  # noqa: E402


def _measure_ms(fn: Callable[[], list[SearchResult]], repeats: int = 300) -> float:
    for _ in range(15):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    idx: Index = build_index(make_pipeline(ROOT_DIR / "data"), positions=True)
    idx.warm_numpy_cache()

    frequent_term = max(idx, key=lambda t: (idx.df(t), t))
    rare_term = next(t for t in idx if idx.df(t) == 1 and len(t) >= 5)
    three_terms = f"{frequent_term} OR люцифер OR рай"

    cases: list[tuple[str, str]] = [
        (f"Частий терм (`{frequent_term}`)", frequent_term),
        (f"Рідкісний терм (`{rare_term}`)", rare_term),
        (f"Запит із 3 термів (`{three_terms}`)", three_terms),
    ]

    print(
        "| Запит | Лаба 3 (цикл Python), мс | NumPy (`int32` + `argpartition`), мс | Прискорення |"
    )
    print("| :--- | :---: | :---: | :---: |")
    for label, q in cases:
        t_leg = _measure_ms(lambda query=q: search_legacy(idx, query, k=10))
        t_np = _measure_ms(lambda query=q: search(idx, query, k=10))
        speedup = t_leg / max(t_np, 1e-6)
        print(f"| {label} | {t_leg:.4f} | {t_np:.4f} | {speedup:.2f}× |")


if __name__ == "__main__":
    main()
