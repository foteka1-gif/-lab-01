"""Ранжований пошук: парсер -> evaluate -> Scorer -> heapq.nlargest."""

import functools
import heapq
import logging
from dataclasses import dataclass, field
from typing import NamedTuple

from .decorators import timed
from .model import Index
from .query import parse
from .scoring import BM25, Scorer

log = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class SearchResult:
    # order=True порівнює поля за порядком оголошення -> score першим,
    # тож sorted(results) / max(results) впорядковують за релевантністю.
    score: float
    doc_id: int
    title: str
    snippet: str = field(default="", compare=False)

    def __str__(self) -> str:
        return f"{self.score:8.4f} | ID {self.doc_id} | {self.title}"


class CacheStats(NamedTuple):
    hits: int
    misses: int
    maxsize: int | None
    currsize: int


class Matcher:
    """Кеш «рядок запиту -> кортеж doc_id» для конкретного індексу.

    Це клас, а не замикання з приліпленими атрибутами: так pyright (strict)
    бачить cache_info/cache_clear як звичайні методи.
    """

    def __init__(self, index: Index, maxsize: int = 256) -> None:
        self._index = index
        self._cached = functools.lru_cache(maxsize=maxsize)(self._evaluate)

    def _evaluate(self, query: str) -> tuple[int, ...]:
        return tuple(sorted(parse(query).evaluate(self._index)))

    def __call__(self, query: str) -> tuple[int, ...]:
        hits_before = self._cached.cache_info().hits
        ids = self._cached(query)
        status = "HIT" if self._cached.cache_info().hits > hits_before else "MISS"
        log.debug("query cache %s: %r -> %d docs", status, query, len(ids))
        return ids

    def cache_info(self) -> CacheStats:
        info = self._cached.cache_info()
        return CacheStats(info.hits, info.misses, info.maxsize, info.currsize)

    def cache_clear(self) -> None:
        self._cached.cache_clear()


@timed
def search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 10,
    matcher: Matcher | None = None,
) -> list[SearchResult]:
    """Повертає top-k SearchResult (найкращі першими)."""
    scorer = scorer if scorer is not None else BM25()
    tree = parse(query)
    matched = set(matcher(query)) if matcher is not None else tree.evaluate(index)

    scores = dict.fromkeys(matched, 0.0)  # документи, що пройшли булевий фільтр
    for term in dict.fromkeys(tree.terms()):  # унікальні, у порядку появи
        for posting in index.get(term, []):
            if posting.doc_id in scores:
                scores[posting.doc_id] += scorer(term, posting, index)

    # O(n log k), а не повне сортування; при рівних балах — менший doc_id
    top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
    return [
        SearchResult(score=score, doc_id=doc_id, title=index.meta(doc_id).title)
        for doc_id, score in top
    ]
