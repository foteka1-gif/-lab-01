"""Один скрипт замість десятка команд: корпус -> бенчмарк -> (Python без GIL) -> графік -> прибирання.

    python scripts/run_all.py                 # звичайний запуск
    python scripts/run_all.py --light         # полегшений (слабкий ноутбук)
    python scripts/run_all.py --nogil         # додатково Python 3.13t (потрібен інтернет для uv)

Що робить для безпеки ноутбука:
  * знижує свій пріоритет (діти успадковують його), тож система лишається чутливою;
  * тимчасовий корпус кладе в папку Temp, а не в проєкт (OneDrive не синхронізує сотні файлів);
  * якщо вільної пам'яті мало, сам прибирає найважчі конфігурації;
  * після роботи видаляє тимчасовий корпус; Ctrl+C безпечний.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PY = sys.executable


def lower_priority() -> None:
    """BELOW_NORMAL на Windows, nice(10) на Unix: система не гальмуватиме."""
    try:
        import psutil

        process = psutil.Process()
        if sys.platform == "win32":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            process.nice(10)
        print("Пріоритет знижено: ноутбук лишається чутливим.")
    except Exception:
        if hasattr(os, "nice"):
            os.nice(10)
        else:
            print("psutil недоступний: пріоритет не змінено (це не страшно).")


def available_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return None


def step(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", **kwargs)  # type: ignore[call-overload]


def main() -> None:
    parser = argparse.ArgumentParser(description="Повний бенчмарк findex однією командою.")
    parser.add_argument("--src", type=Path, default=ROOT / "data", help="папка з книгами")
    parser.add_argument("--copies", type=int, default=8, help="скільки разів повторити набір")
    parser.add_argument("--repeats", type=int, default=3, help="запусків на конфігурацію")
    parser.add_argument("--light", action="store_true", help="полегшений режим")
    parser.add_argument("--nogil", action="store_true", help="додати Python 3.13t (потрібен uv)")
    parser.add_argument("--keep-corpus", action="store_true", help="не видаляти тимчасовий корпус")
    args = parser.parse_args()

    copies = 4 if args.light else args.copies
    cpu_count = os.cpu_count() or 1
    workers = [1, 2, 4, 8] if args.light else sorted({1, 2, 4, 8, cpu_count})

    lower_priority()
    free = available_gb()
    if free is not None:
        print(f"Вільної оперативної пам'яті: {free:.1f} ГБ")
        if free < 3.0 and len(workers) > 3:
            workers = [1, 2, 4]
            print("Пам'яті мало: вимірюю лише 1, 2, 4 воркери (напиши про це в README).")

    if not list(args.src.glob("*.txt")):
        raise SystemExit(f"У {args.src} немає .txt книг.")

    corpus = Path(tempfile.gettempdir()) / "findex_bench_corpus"
    DOCS.mkdir(exist_ok=True)
    worker_args = ["--workers", *map(str, workers)]

    try:
        step(f"1/5 Корпус у {corpus} (копій: {copies})")
        shutil.rmtree(corpus, ignore_errors=True)
        run(
            [
                PY,
                "scripts/make_corpus.py",
                str(args.src),
                str(corpus),
                "--chars",
                "50000",
                "--copies",
                str(copies),
            ],
            check=True,
        )

        step(f"2/5 Бенчмарк (воркери: {workers}, запусків: {args.repeats}) — кілька хвилин")
        run(
            [
                PY,
                "scripts/bench_index.py",
                str(corpus),
                "--repeats",
                str(args.repeats),
                *worker_args,
                "--out",
                "results.json",
            ],
            check=True,
        )

        results = ["results.json"]
        step("3/5 Досліди GIL і гонитви (звичайний Python)")
        for name, command in (
            ("gil_demo.txt", ["experiments/gil_demo.py"]),
            ("race_gil.txt", ["experiments/race_demo.py", "--fast-switch"]),
        ):
            done = run([PY, *command], capture_output=True)
            (DOCS / name).write_text(done.stdout, encoding="utf-8")
            print(f"  збережено docs/{name}")

        if args.nogil:
            step("4/5 Python 3.13t (без GIL)")
            if shutil.which("uv") is None:
                print("  uv не знайдено: цей крок пропущено.")
            elif run(["uv", "python", "install", "3.13t"]).returncode != 0:
                print("  Не вдалося встановити 3.13t (немає інтернету?): крок пропущено.")
            else:
                env = {
                    **os.environ,
                    "UV_PROJECT_ENVIRONMENT": str(Path(tempfile.gettempdir()) / "findex_venv313t"),
                }
                base = ["uv", "run", "--frozen", "--python", "3.13t", "--no-dev"]
                bench = [
                    "python",
                    "scripts/bench_index.py",
                    str(corpus),
                    "--repeats",
                    str(args.repeats),
                    *worker_args,
                    "--executors",
                    "serial",
                    "threads",
                    "--out",
                    "results_313t.json",
                ]
                done = run([*base, "--with", "psutil", *bench], env=env)
                if done.returncode != 0:
                    print("  psutil не встановився: пік пам'яті в цьому файлі буде 0.")
                    done = run([*base, *bench], env=env)
                if done.returncode == 0:
                    results.append("results_313t.json")
                    for name, command in (
                        ("race_313t.txt", ["experiments/race_demo.py"]),
                        ("gil_demo_313t.txt", ["experiments/gil_demo.py"]),
                    ):
                        out = run([*base, "python", *command], env=env, capture_output=True)
                        (DOCS / name).write_text(out.stdout, encoding="utf-8")
                        print(f"  збережено docs/{name}")
                else:
                    print("  Замір на 3.13t не вдався: рядків без GIL не буде.")
        else:
            step("4/5 Python 3.13t пропущено (додай --nogil, коли основне готове)")

        step("5/5 Таблиця і графік")
        done = run(
            [PY, "scripts/plot_speedup.py", *results, "--out", "docs/speedup.png"],
            capture_output=True,
        )
        (DOCS / "results_table.md").write_text(done.stdout, encoding="utf-8")
        print(done.stdout)
        if done.returncode != 0:
            print(done.stderr)
    finally:
        if not args.keep_corpus:
            shutil.rmtree(corpus, ignore_errors=True)
            print("Тимчасовий корпус видалено.")

    print("\nГОТОВО. Дивись: docs/speedup.png, docs/results_table.md, results.json")


if __name__ == "__main__":
    main()
