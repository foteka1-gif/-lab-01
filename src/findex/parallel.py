"""Паралельна побудова індексу: розбити шляхи на шматки -> build_partial у воркерах -> merge.

Три виконавці з однаковим кодом: serial (звичайний цикл), threads (ThreadPoolExecutor),
processes (ProcessPoolExecutor). Змінюється лише те, ДЕ виконується build_partial.

Імпортує лише «легкі» модулі (без typer/rich): метод spawn заново імпортує цей модуль
у кожному воркері, і чим менше він тягне за собою, тим швидший старт.
"""

import logging
import multiprocessing
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from .index import PartialIndex, build_partial, merge
from .model import Index

log = logging.getLogger(__name__)

type ExecutorName = Literal["serial", "threads", "processes"]


@dataclass(frozen=True, slots=True)
class Chunk:
    """Шматок роботи: неперервна частина відсортованого списку шляхів + її діапазон doc_id."""

    paths: tuple[Path, ...]
    first_doc_id: int


class ChunkResult(NamedTuple):
    partial: PartialIndex
    pid: int  # у якому процесі виконано шматок
    process_cpu: float  # скільки CPU-секунд цей процес витратив від свого старту


class BuildReport(NamedTuple):
    built_index: Index
    wall_seconds: float  # скільки минуло на годиннику
    cpu_seconds: float  # скільки процесорного часу витрачено ВСІМА процесами
    merge_seconds: float  # час merge (послідовна частина)
    chunks: int


def split_paths(paths: Sequence[Path], parts: int) -> list[Chunk]:
    """Ділить шляхи на `parts` неперерних шматків, що відрізняються розміром не більше ніж на 1.

    first_doc_id шматка = індекс його першого шляху в загальному списку, тому doc_id
    унікальні на весь корпус і збігаються з тими, що дає послідовна збірка.
    """
    parts = max(1, min(parts, len(paths)))
    base, extra = divmod(len(paths), parts)
    chunks: list[Chunk] = []
    start = 0
    for i in range(parts):
        size = base + (1 if i < extra else 0)
        chunks.append(Chunk(tuple(paths[start : start + size]), first_doc_id=start))
        start += size
    return chunks


def run_chunk(chunk: Chunk, positions: bool) -> ChunkResult:
    """Те, що виконує воркер. Функція рівня модуля: її можна відправити в інший процес."""
    partial = build_partial(chunk.paths, chunk.first_doc_id, positions)
    return ChunkResult(partial, os.getpid(), time.process_time())


def _make_pool(executor: Literal["threads", "processes"], workers: int) -> Executor:
    if executor == "threads":
        return ThreadPoolExecutor(max_workers=workers)
    # spawn однаково працює на Windows, macOS і Linux (і саме він найстрогіший до коду).
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def _run_chunks(
    plan: Sequence[Chunk],
    workers: int,
    executor: ExecutorName,
    positions: bool,
    progress: Callable[[int], None] | None,
) -> list[ChunkResult]:
    results: list[ChunkResult] = []

    if executor == "serial":
        for chunk in plan:
            results.append(run_chunk(chunk, positions))
            if progress is not None:
                progress(len(chunk.paths))
        return results

    with _make_pool(executor, workers) as pool:
        futures: dict[Future[ChunkResult], Chunk] = {
            pool.submit(run_chunk, chunk, positions): chunk for chunk in plan
        }
        try:
            for future in as_completed(futures):
                # result() знову кидає виняток воркера (з його стеком): збірка падає, а не висить.
                results.append(future.result())
                if progress is not None:
                    progress(len(futures[future].paths))
        except BaseException:
            for future in futures:
                future.cancel()  # ще не запущені шматки скасовуємо
            raise
    return results


def _workers_cpu(results: Sequence[ChunkResult]) -> float:
    """CPU-секунди процесів-воркерів: для кожного pid береться останнє (найбільше) значення."""
    parent = os.getpid()
    per_process: dict[int, float] = {}
    for result in results:
        if result.pid != parent:
            per_process[result.pid] = max(per_process.get(result.pid, 0.0), result.process_cpu)
    return sum(per_process.values())


def build_index_parallel(
    paths: Sequence[Path],
    workers: int = 1,
    executor: ExecutorName = "serial",
    positions: bool = False,
    chunks: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> BuildReport:
    """Будує індекс по шматках.

    chunks за замовчуванням: 1 для serial (= merge([build_partial(усі шляхи)])), інакше
    стільки, скільки воркерів. progress(n) викликається в головному потоці після кожного
    готового шматка, n = кількість документів у ньому.
    """
    if workers < 1:
        raise ValueError("workers має бути не менше 1.")
    if chunks is not None and chunks < 1:
        raise ValueError("chunks має бути не менше 1.")

    n_chunks = chunks if chunks is not None else (1 if executor == "serial" else workers)
    plan = split_paths(paths, n_chunks)
    log.info("executor=%s workers=%d chunks=%d docs=%d", executor, workers, len(plan), len(paths))

    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    results = _run_chunks(plan, workers, executor, positions, progress)

    merge_start = time.perf_counter()
    index = merge(result.partial for result in results)
    merge_seconds = time.perf_counter() - merge_start

    wall_seconds = time.perf_counter() - wall_start
    # CPU головного процесу (у ньому живуть усі потоки) + CPU воркерів-процесів.
    cpu_seconds = time.process_time() - cpu_start + _workers_cpu(results)

    log.info(
        "wall=%.3f s cpu=%.3f s merge=%.3f s",
        wall_seconds,
        cpu_seconds,
        merge_seconds,
    )
    return BuildReport(index, wall_seconds, cpu_seconds, merge_seconds, len(plan))
