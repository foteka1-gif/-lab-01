"""«Prove it to yourself», дослід 1: чи прискорюють потоки і процеси чисту роботу процесора?

python experiments/gil_demo.py
uv run --python 3.13t python experiments/gil_demo.py     # без GIL
"""

import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

N = 10**7


def work() -> int:
    return sum(i * i for i in range(N))


def measure(label: str, run: Callable[[], object]) -> None:
    start = time.perf_counter()
    run()
    print(f"{label:34} {time.perf_counter() - start:6.2f} с")


def run_in(pool_type: type[ThreadPoolExecutor] | type[ProcessPoolExecutor]) -> None:
    with pool_type(max_workers=2) as pool:
        futures = [pool.submit(work), pool.submit(work)]
        for future in futures:
            future.result()


def main() -> None:
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"Python {sys.version.split()[0]}, GIL увімкнено: {gil}\n")

    measure("1 виклик", work)
    measure("2 виклики підряд", lambda: (work(), work()))
    measure("2 потоки", lambda: run_in(ThreadPoolExecutor))
    measure("2 процеси", lambda: run_in(ProcessPoolExecutor))


if __name__ == "__main__":  # обов'язково: процеси на Windows заново імпортують цей файл
    main()
