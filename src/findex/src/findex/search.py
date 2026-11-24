"""Ранжований пошук: парсер -> evaluate -> векторний Scorer (NumPy) -> np.argpartition."""

import functools
import heapq
import logging
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .decorators import timed
from .model import Index
from .query import parse
from .scoring import BM25, Scorer, VectorScorer

log = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class SearchResult:
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
    """Кеш «рядок запиту -> кортеж doc_id» для конкретного індексу."""

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


def search_legacy(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 10,
    matcher: Matcher | None = None,
) -> list[SearchResult]:
    """Базовий варіант із лаби 3 (цикл Python + heapq.nlargest) для порівняння швидкості."""
    scorer = scorer if scorer is not None else BM25()
    tree = parse(query)
    matched = set(matcher(query)) if matcher is not None else tree.evaluate(index)

    scores = dict.fromkeys(matched, 0.0)
    for term in dict.fromkeys(tree.terms()):
        for posting in index.get(term, []):
            if posting.doc_id in scores:
                scores[posting.doc_id] += scorer(term, posting, index)

    top = heapq.nlargest(k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
    return [
        SearchResult(score=score, doc_id=doc_id, title=index.meta(doc_id).title)
        for doc_id, score in top
    ]


@timed
def search(
    index: Index,
    query: str,
    scorer: Scorer | None = None,
    k: int = 10,
    matcher: Matcher | None = None,
) -> list[SearchResult]:
    """Повертає top-k SearchResult за допомогою NumPy-векторизації та np.argpartition."""
    if k <= 0 or index.num_docs == 0:
        return []

    active_scorer = scorer if scorer is not None else BM25()
    if not isinstance(active_scorer, VectorScorer):
        return search_legacy(index, query, scorer=active_scorer, k=k, matcher=matcher)

    tree = parse(query)
    matched_seq = matcher(query) if matcher is not None else tree.evaluate(index)
    if not matched_seq:
        return []

    size = index.max_doc_id + 1
    scores = np.zeros(size, dtype=np.float64)

    for term in dict.fromkeys(tree.terms()):
        doc_ids, term_scores = active_scorer.score_array(term, index)
        if doc_ids.size > 0:
            scores[doc_ids] += term_scores

    matched_ids = np.fromiter(matched_seq, dtype=np.int32, count=len(matched_seq))
    matched_scores = scores[matched_ids]
    n_matched = int(matched_ids.size)
    top_k = min(k, n_matched)

    # Розв'язання нічиїх ідентичне лабі 3: вищий score вище, при рівних балах — менший doc_id.
    # np.argpartition знаходить індекси top_k найменших ключів за O(n).
    order_keys = np.lexsort((matched_ids, -matched_scores)) if n_matched <= 64 else None
    if order_keys is not None:
        chosen = order_keys[:top_k]
    else:
        part_idx = np.argpartition(-matched_scores, kth=top_k - 1)[:top_k]
        sub_order = np.lexsort((matched_ids[part_idx], -matched_scores[part_idx]))
        chosen = part_idx[sub_order]

    top_doc_ids = matched_ids[chosen]
    top_scores = matched_scores[chosen]
    return [
        SearchResult(
            score=float(top_scores[i]),
            doc_id=int(top_doc_ids[i]),
            title=index.meta(int(top_doc_ids[i])).title,
        )
        for i in range(top_k)
    ]
