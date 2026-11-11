"""Ранжований пошук: парсер -> evaluate -> Scorer -> heapq.nlargest -> сніпети."""
import argparse
import functools
import heapq
import time
from dataclasses import dataclass, field
 
from .decorators import timed, setup_logging, log
from .model import open_index
from .query import parse
from .scoring import BM25, TfIdf
from .snippets import snippet_for
 
 
@dataclass(frozen=True, order=True)
class SearchResult:
    # order=True порівнює поля за порядком оголошення -> score першим,
    # тож sorted(results) / max(results) впорядковують за релевантністю.
    score: float
    doc_id: int
    title: str
    snippet: str = field(default="", compare=False)
 
    def __str__(self):
        return f"{self.score:8.4f} | ID {self.doc_id} | {self.title}"
 
 
@timed
def search(index, query, scorer=None, k=10, matcher=None):
    """Повертає top-k SearchResult (найкращі першими)."""
    scorer = scorer if scorer is not None else BM25()
    tree = parse(query)
    matched = set(matcher(query)) if matcher else tree.evaluate(index)
 
    scores = dict.fromkeys(matched, 0.0)  # документи, що пройшли булевий фільтр
    for term in dict.fromkeys(tree.terms()):  # унікальні, у порядку появи
        for posting in index.get(term, ()):
            if posting.doc_id in scores:
                scores[posting.doc_id] += scorer(term, posting, index)
 
    # O(n log k), а не повне сортування O(n log n); при рівних балах — менший doc_id
    top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
    return [
        SearchResult(score=score, doc_id=doc_id, title=index.meta(doc_id).title)
        for doc_id, score in top
    ]
 
 
def make_matcher(index, maxsize=256):
    """Замикання: lru_cache живе поруч з конкретним index (сам Index не хешується).
    Кеш: рядок запиту -> кортеж doc_id."""
 
    @functools.lru_cache(maxsize=maxsize)
    def cached(query):
        return tuple(sorted(parse(query).evaluate(index)))
 
    def matcher(query):
        hits_before = cached.cache_info().hits
        ids = cached(query)
        status = "HIT" if cached.cache_info().hits > hits_before else "MISS"
        log.debug("query cache %s: %r -> %d docs", status, query, len(ids))
        return ids
 
    matcher.cache_info = cached.cache_info
    matcher.cache_clear = cached.cache_clear
    return matcher
 
 
def main():
    parser = argparse.ArgumentParser(description="Ранжований пошук за індексом.")
    parser.add_argument("index_file")
    parser.add_argument("query")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle")
    parser.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1,
                        help="виконати запит N разів (для демонстрації cache hit)")
    parser.add_argument("--no-snippets", action="store_true")
    args = parser.parse_args()
 
    setup_logging()
    scorer = BM25() if args.scorer == "bm25" else TfIdf()
 
    try:
        with open_index(args.index_file, args.format) as index:
            print(index)
            matcher = make_matcher(index)
            terms = parse(args.query).terms()
 
            for _ in range(args.repeat):
                start = time.perf_counter()
                results = search(index, args.query, scorer, args.k, matcher)
                elapsed = time.perf_counter() - start
 
            print("\nРЕЗУЛЬТАТИ ПОШУКУ")
            print("-------------------------")
            print("Запит:", args.query)
            print("Scorer:", scorer)
            print("Знайдено (top-k):", len(results))
            print(f"Час останнього пошуку: {elapsed:.6f} сек.")
            print("Кеш:", matcher.cache_info())
 
            for rank, result in enumerate(results, 1):
                print(f"\n{rank}. {result}")
                if not args.no_snippets:
                    print("   ", snippet_for(index.meta(result.doc_id), terms))
    except ValueError as error:
        parser.error(str(error))
 
 
if __name__ == "__main__":
    main()