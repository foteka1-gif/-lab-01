"""Порівняння пам'яті (dataclass / slots / array) та merge проти set.

Функції повертають дані, а не друкують: виводить їх той, хто викликає.
"""

import time
import tracemalloc
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

from . import Posting
from .boolean import merge_and, set_operation


@dataclass(frozen=True)
class NormalPosting:
    doc_id: int
    tf: int


class MemoryReport(NamedTuple):
    postings: int
    normal_mb: float
    slots_mb: float
    array_mb: float


class SearchTiming(NamedTuple):
    label: str
    terms: tuple[str, str]
    merge_seconds: float
    set_seconds: float


def measure_factory_memory(factory: Callable[[], object]) -> float:
    tracemalloc.start()
    data = factory()
    _current, peak = tracemalloc.get_traced_memory()
    del data
    tracemalloc.stop()
    return peak / 1024 / 1024


def measure_memory_all(postings: Mapping[str, Sequence[Posting]]) -> MemoryReport:
    # postings може бути і dict, і Index: потрібен лише .values()
    all_items = [(p.doc_id, p.tf) for items in postings.values() for p in items]

    def normal_factory() -> list[NormalPosting]:
        return [NormalPosting(doc_id, tf) for doc_id, tf in all_items]

    def slots_factory() -> list[Posting]:
        return [Posting(doc_id, tf) for doc_id, tf in all_items]

    def array_factory() -> tuple[array[int], array[int]]:
        doc_ids = array("I")
        frequencies = array("I")
        for doc_id, tf in all_items:
            doc_ids.append(doc_id)
            frequencies.append(tf)
        return doc_ids, frequencies

    return MemoryReport(
        postings=len(all_items),
        normal_mb=measure_factory_memory(normal_factory),
        slots_mb=measure_factory_memory(slots_factory),
        array_mb=measure_factory_memory(array_factory),
    )


def benchmark_search(
    postings: Mapping[str, Sequence[Posting]], repeats: int = 1000
) -> list[SearchTiming]:
    if len(postings) < 2:
        return []

    sorted_terms = sorted(postings, key=lambda term: sum(p.tf for p in postings[term]))
    tests = [("Найрідкісніші", sorted_terms[:2]), ("Найчастіші", sorted_terms[-2:])]

    rows: list[SearchTiming] = []
    for label, terms in tests:
        a = postings[terms[0]]
        b = postings[terms[1]]

        start = time.perf_counter()
        for _ in range(repeats):
            merge_and(a, b)
        merge_time = (time.perf_counter() - start) / repeats

        start = time.perf_counter()
        for _ in range(repeats):
            set_operation(a, b, "AND")
        set_time = (time.perf_counter() - start) / repeats

        rows.append(SearchTiming(label, (terms[0], terms[1]), merge_time, set_time))
    return rows
