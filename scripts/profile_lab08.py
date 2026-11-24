"""Профілювання findex index та findex search (cProfile + генерація SVG flamegraph)."""

import cProfile
import io
import pstats
import sys
from collections.abc import Callable
from html import escape
from pathlib import Path
from typing import cast

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from findex.index import build_index, make_pipeline  # noqa: E402
from findex.model import Index  # noqa: E402
from findex.search import search  # noqa: E402

type StatRow = tuple[str, float, float, int]
type RawStatsDict = dict[tuple[str, int, str], tuple[int, int, float, float, object]]


def _run_and_stats(fn: Callable[[], None], label: str) -> list[StatRow]:
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()

    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream).sort_stats("tottime")
    stats.print_stats(10)
    print(f"\n=== cProfile ({label}) ===")
    print(stream.getvalue())

    raw_stats = cast(RawStatsDict, getattr(stats, "stats", {}))
    rows: list[StatRow] = []
    for (filename, line, func_name), (cc, _nc, tt, ct, _callers) in raw_stats.items():
        short_file = Path(filename).name
        rows.append((f"{short_file}:{line} ({func_name})", float(tt), float(ct), int(cc)))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:10]


def _write_flamegraph_svg(
    out_path: Path,
    index_rows: list[StatRow],
    search_rows: list[StatRow],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width = 960
    bar_h = 26
    gap = 6

    def render_section(title: str, rows: list[StatRow], y_offset: int) -> tuple[str, int]:
        max_tt = max((r[1] for r in rows), default=1e-6) or 1e-6
        palette = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#8b5cf6"]
        lines = [
            f'<text x="20" y="{y_offset}" font-size="16" font-weight="700" fill="#0f172a">{escape(title)}</text>'
        ]
        y = y_offset + 14
        for idx, (name, tt, ct, calls) in enumerate(rows[:8]):
            w = max(8, int((tt / max_tt) * (width - 260)))
            color = palette[idx % len(palette)]
            label = (
                f"{name} — tottime: {tt * 1000:.2f} ms, cumtime: {ct * 1000:.2f} ms ({calls} calls)"
            )
            lines.append(
                f'<rect x="20" y="{y}" width="{w}" height="{bar_h}" rx="4" fill="{color}" opacity="0.88" />'
            )
            lines.append(
                f'<text x="28" y="{y + 17}" font-size="12" font-family="monospace" fill="#0f172a">{escape(label)}</text>'
            )
            y += bar_h + gap
        return "\n".join(lines), y + 20

    sec1, next_y = render_section(
        "Flame Graph (tottime): findex index (build_index)", index_rows, 30
    )
    sec2, total_h = render_section(
        "Flame Graph (tottime): findex search (3 terms, 200 runs)", search_rows, next_y
    )

    svg = (
        f'<svg xmlns="http://w3.org" width="{width}" height="{total_h}" '
        f'viewBox="0 0 {width} {total_h}" style="background:#ffffff;font-family:system-ui,sans-serif;">\n'
        f"{sec1}\n{sec2}\n</svg>\n"
    )
    out_path.write_text(svg, encoding="utf-8")


def main() -> None:
    data_dir = ROOT_DIR / "data"
    built_index: list[Index] = []

    def do_index() -> None:
        idx = build_index(make_pipeline(data_dir), positions=True)
        built_index.append(idx)

    index_rows = _run_and_stats(do_index, "findex index")
    idx = next(iter(built_index))

    def do_search() -> None:
        for _ in range(200):
            search(idx, "ангел OR люцифер OR рай", k=10)

    search_rows = _run_and_stats(do_search, "findex search (3 terms x 200)")
    svg_path = ROOT_DIR / "docs" / "flamegraph.svg"
    _write_flamegraph_svg(svg_path, index_rows, search_rows)
    print(f"Flame graph збережено у {svg_path}")


if __name__ == "__main__":
    main()
