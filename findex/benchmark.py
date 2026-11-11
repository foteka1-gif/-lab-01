import time
import tracemalloc
from array import array
from dataclasses import dataclass
 
from . import Posting
from .boolean import merge_and, set_operation  # ЛАБ 3: було `from .search import ...`
 
 
@dataclass(frozen=True)
class NormalPosting:
    doc_id: int
    tf: int
 
 
def measure_factory_memory(factory):
    tracemalloc.start()
    data = factory()
    current, peak = tracemalloc.get_traced_memory()
    del data
    tracemalloc.stop()
    return peak / 1024 / 1024
 
 
def measure_memory_all(postings):
    # postings може бути і dict, і Index: потрібен лише .values()
    all_items = [(p.doc_id, p.tf) for items in postings.values() for p in items]
 
    def normal_factory():
        return [NormalPosting(doc_id, tf) for doc_id, tf in all_items]
 
    def slots_factory():
        return [Posting(doc_id, tf) for doc_id, tf in all_items]
 
    def array_factory():
        doc_ids = array("I")
        frequencies = array("I")
        for doc_id, tf in all_items:
            doc_ids.append(doc_id)
            frequencies.append(tf)
        return doc_ids, frequencies
 
    print("\nПОРІВНЯННЯ ПАМ'ЯТІ")
    print("-------------------------")
    print("Кількість постінгів:", len(all_items))
 
    normal = measure_factory_memory(normal_factory)
    slots = measure_factory_memory(slots_factory)
    arrays = measure_factory_memory(array_factory)
 
    print(f"Звичайний dataclass: {normal:.4f} MB")
    print(f"dataclass slots=True: {slots:.4f} MB")
    print(f"array('I'): {arrays:.4f} MB")
    print(f"Економія slots: {normal - slots:.4f} MB")
    print(f"Економія array: {normal - arrays:.4f} MB")
 
 
def benchmark_search(postings):
    print("\nПОРІВНЯННЯ MERGE ТА SET")
    print("-------------------------")
 
    if len(postings) < 2:
        print("Недостатньо слів для порівняння.")
        return
 
    sorted_terms = sorted(postings, key=lambda term: sum(p.tf for p in postings[term]))
    tests = [("Найрідкісніші", sorted_terms[:2]), ("Найчастіші", sorted_terms[-2:])]
    repeats = 1000
 
    for label, terms in tests:
        if len(terms) < 2:
            continue
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
 
        print(f"\n{label}")
        print("Слова:", terms[0], "і", terms[1])
        print(f"Merge: {merge_time:.8f} сек.")
        print(f"Set: {set_time:.8f} сек.")
 
 
def measure_memory_report(postings):
    measure_memory_all(postings)
