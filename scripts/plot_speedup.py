"""Таблиця (markdown) і графік прискорення з одного чи кількох results*.json.

    python scripts/plot_speedup.py results.json results_313t.json --out docs/speedup.png

Графік малюється окремо від вимірювань: на Python 3.13t matplotlib може не ставитися,
а вимірювати треба саме там.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def label_for(result: dict[str, Any], executor: str) -> str:
    machine = result["machine"]
    suffix = "" if machine["gil_enabled"] else f" (Python {machine['python']}, без GIL)"
    return f"{executor}{suffix}"


def print_table(results: list[dict[str, Any]]) -> None:
    print(
        "| Виконавець | Воркери | Wall, с (медіана) | CPU, с | Peak RSS, МБ | Merge, с | Прискорення |"
    )
    print("|---|---|---|---|---|---|---|")
    for result in results:
        for row in result["rows"]:
            name = label_for(result, row["executor"])
            print(
                f"| {name} | {row['workers']} | {row['wall']:.2f} | {row['cpu']:.2f} | "
                f"{row['rss_mb']:.0f} | {row['merge']:.3f} | {row['speedup']:.2f}× |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Графік прискорення.")
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("docs/speedup.png"))
    args = parser.parse_args()

    results: list[dict[str, Any]] = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.results
    ]
    print_table(results)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("Для графіка потрібен matplotlib: uv add --dev matplotlib") from None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    all_workers: set[int] = set()
    for result in results:
        for executor in ("threads", "processes"):
            points = sorted(
                (row["workers"], row["speedup"])
                for row in result["rows"]
                if row["executor"] == executor
            )
            if points:
                all_workers.update(w for w, _ in points)
                ax.plot(*zip(*points, strict=False), marker="o", label=label_for(result, executor))

    top = max(all_workers, default=1)
    ax.plot([1, top], [1, top], "k--", label="ідеальний лінійний ріст")
    ax.set_xlabel("Число воркерів")
    ax.set_ylabel("Прискорення (serial wall / wall)")
    ax.set_title("Прискорення побудови індексу")
    ax.grid(True, alpha=0.3)
    ax.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nГрафік збережено: {args.out}")


if __name__ == "__main__":
    main()
