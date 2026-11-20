"""Гонитва даних: counter += 1 не атомарний.

    python experiments/race_demo.py
    python experiments/race_demo.py --fast-switch        # частіші перемикання потоків
    uv run --python 3.13t python experiments/race_demo.py  # без GIL: втрати видно одразу

Що очікувати. З GIL прості `shared += 1` у коротких циклах майже не губляться (перемикання
рідкісні), але варіант 1б (виклик функції між читанням і записом) з --fast-switch губить
оновлення і з GIL. Без GIL (3.13t) втрати видно в 1), 1б) і 4) одразу. Варіанти 2) і 3)
правильні всюди: Lock або відсутність спільного стану.
"""

import argparse
import dis
import sys
import threading
from collections import Counter
from collections.abc import Callable

N = 1_000_000
shared = 0
lock = threading.Lock()


def bump_unsafe(n: int) -> None:
    global shared
    for _ in range(n):
        shared += 1  # читання -> додавання -> запис: між ними може перемкнутись потік


def _noop() -> None:
    pass


def bump_unsafe_slow(n: int) -> None:
    """Те саме, але читання й запис рознесені викликом функції.

    Виклик функції — місце, де інтерпретатор перевіряє, чи не пора віддати GIL. Тож навіть
    зі звичайним GIL тут оновлення губляться: GIL захищає ІНТЕРПРЕТАТОР, а не твій інваріант.
    """
    global shared
    for _ in range(n):
        current = shared
        _noop()
        shared = current + 1


def bump_locked(n: int) -> None:
    global shared
    for _ in range(n):
        with lock:
            shared += 1


def count_local(n: int, results: list[int]) -> None:
    local = 0  # власний стан потоку: ділити нічого
    for _ in range(n):
        local += 1
    results.append(local)


WORDS = ["a", "b", "c", "d"] * 50_000
shared_counts: Counter[str] = Counter()


def count_words_unsafe(words: list[str]) -> None:
    for word in words:
        shared_counts[word] += 1  # той самий read-modify-write, тільки по ключу


def run_threads(target: Callable[..., None], *args: object, threads: int = 2) -> None:
    workers = [threading.Thread(target=target, args=args) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


def main() -> None:
    global shared
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-switch", action="store_true")
    args = parser.parse_args()
    if args.fast_switch:
        sys.setswitchinterval(1e-6)  # за замовчуванням 5 мс

    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"Python {sys.version.split()[0]}, GIL увімкнено: {gil}\n")

    print("Байткод `shared += 1` (потік може перемкнутись між будь-якими двома рядками):")
    dis.dis(bump_unsafe)

    expected = 2 * N
    shared = 0
    run_threads(bump_unsafe, N)
    print(f"\n1) без Lock:  {shared:>9,} із {expected:,}  (втрачено {expected - shared:,})")

    shared = 0
    run_threads(bump_unsafe_slow, N // 10)
    expected_slow = 2 * (N // 10)
    print(
        f"1б) з викликом між читанням і записом: {shared:,} із {expected_slow:,}"
        f"  (втрачено {expected_slow - shared:,})"
    )

    shared = 0
    run_threads(bump_locked, N)
    print(f"2) з Lock:    {shared:>9,} із {expected:,}")

    results: list[int] = []
    run_threads(count_local, N, results)
    print(f"3) без спільного стану: {sum(results):,} із {expected:,}  (сума після join)")

    run_threads(count_words_unsafe, WORDS)
    total = sum(shared_counts.values())
    print(f"4) спільний Counter без Lock: {total:,} із {2 * len(WORDS):,}")


if __name__ == "__main__":
    main()
