import argparse
import time
# Чисті відносні імпорти
from .store import load_pickle, load_json
from . import Posting

def merge_and(list_a: list[Posting], list_b: list[Posting]) -> list[Posting]:
    """
    Алгоритм двопокажчикового злиття для логічного AND.
    Складність: O(len(a) + len(b))
    """
    result = []
    i, j = 0, 0
    
    while i < len(list_a) and j < len(list_b):
        if list_a[i].doc_id == list_b[j].doc_id:
            result.append(Posting(doc_id=list_a[i].doc_id, tf=list_a[i].tf + list_b[j].tf))
            i += 1
            j += 1
        elif list_a[i].doc_id < list_b[j].doc_id:
            i += 1
        else:
            j += 1
            
    return result

def search_set_and(list_a: list[Posting], list_b: list[Posting]) -> list[Posting]:
    """
    Альтернативний рушій пошуку на основі вбудованих множин set.
    """
    set_a = {p.doc_id for p in list_a}
    set_b = {p.doc_id for p in list_b}
    common_ids = set_a & set_b
    return [p for p in list_a if p.doc_id in common_ids]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Булевий пошук за індексом.")
    parser.add_argument("index_file", type=str, help="Шлях до файлу індексу")
    parser.add_argument("query", type=str, help="Пошуковий запит")
    parser.add_argument("--format", type=str, default="pickle", choices=["pickle", "json"], help="Формат індексу")
    parser.add_argument("--engine", type=str, default="merge", choices=["merge", "set"], help="Рушій пошуку")
    args = parser.parse_args()
    
    print(f"Завантажуємо індекс з {args.index_file}...")
    if args.format == "pickle":
        postings, doc_lengths, doc_meta = load_pickle(args.index_file)
    else:
        postings, doc_lengths, doc_meta = load_json(args.index_file)
        
    words = args.query.lower().split()
    if not words:
        print("Порожній запит.")
        exit()
        
    print(f"Пошук: {' AND '.join(words)} (рушій: {args.engine})...")
    start_time = time.time()
    
    current_result = postings.get(words[0], [])
    
    for word in words[1:]:
        next_postings = postings.get(word, [])
        if args.engine == "merge":
            current_result = merge_and(current_result, next_postings)
        else:
            current_result = search_set_and(current_result, next_postings)
            
    search_time = time.time() - start_time
    print(f"Знайдено документів: {len(current_result)} за {search_time:.6f} сек.")
    
    print("\nРезультати пошуку:")
    for posting in current_result:
        meta = doc_meta[posting.doc_id]
        print(f"  [ID: {posting.doc_id}] {meta.title} (Файл: {meta.path}, tf={posting.tf})")
