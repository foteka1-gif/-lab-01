"""Старі алгоритми лаб 2 (merge та set). Їх більше не викликає пошук —
пошук тепер іде через query.py, — але benchmark.py досі порівнює їх."""
from . import Posting
 
 
def merge_and(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i].doc_id == b[j].doc_id:
            result.append(Posting(doc_id=a[i].doc_id, tf=a[i].tf + b[j].tf))
            i += 1
            j += 1
        elif a[i].doc_id < b[j].doc_id:
            i += 1
        else:
            j += 1
    return result
 
 
def merge_or(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i].doc_id == b[j].doc_id:
            result.append(Posting(doc_id=a[i].doc_id, tf=a[i].tf + b[j].tf))
            i += 1
            j += 1
        elif a[i].doc_id < b[j].doc_id:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
 
 
def merge_not(a, universe):
    excluded = {p.doc_id for p in a}
    return [Posting(doc_id=p.doc_id, tf=0) for p in universe if p.doc_id not in excluded]
 
 
def set_operation(a, b, operation):
    dict_a = {p.doc_id: p.tf for p in a}
    dict_b = {p.doc_id: p.tf for p in b}
    if operation == "AND":
        ids = dict_a.keys() & dict_b.keys()
    elif operation == "OR":
        ids = dict_a.keys() | dict_b.keys()
    else:
        ids = dict_a.keys() - dict_b.keys()
    return [
        Posting(doc_id=doc_id, tf=dict_a.get(doc_id, 0) + dict_b.get(doc_id, 0))
        for doc_id in sorted(ids)
    ]