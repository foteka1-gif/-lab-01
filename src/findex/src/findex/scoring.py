"""Scorer-протокол + векторні TF-IDF та BM25 на масивах NumPy."""

import math
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from . import Posting
from .model import Index, Int32Array

type ScorerName = Literal["bm25", "tfidf"]
type Float64Array = NDArray[np.float64]


class Scorer(Protocol):
    def __call__(self, term: str, posting: Posting, index: Index) -> float: ...


@runtime_checkable
class VectorScorer(Protocol):
    """Протокол для векторного підрахунку оцінок по всьому списку постінгів терма."""

    def score_array(self, term: str, index: Index) -> tuple[Int32Array, Float64Array]: ...


class TfIdf:
    """(1 + ln tf) * ln(N / df). Терміни, що є в усіх документах, мають вагу 0."""

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf <= 0:
            return 0.0
        idf = math.log(index.num_docs / df)
        return (1.0 + math.log(posting.tf)) * idf

    def score_array(self, term: str, index: Index) -> tuple[Int32Array, Float64Array]:
        doc_ids, tfs = index.posting_arrays(term)
        df = int(doc_ids.size)
        if df == 0:
            return doc_ids, np.empty(0, dtype=np.float64)
        idf = math.log(index.num_docs / df)
        scores = np.zeros(df, dtype=np.float64)
        pos = tfs > 0
        if np.any(pos):
            scores[pos] = (1.0 + np.log(tfs[pos].astype(np.float64))) * idf
        return doc_ids, scores

    def __repr__(self) -> str:
        return "TfIdf()"


class BM25:
    """TF-IDF з насиченням TF (k1) та нормалізацією довжини документа (b)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf <= 0:
            return 0.0
        n = index.num_docs
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        avgdl = index.avg_doc_length or 1.0
        dl = index.doc_length(posting.doc_id)
        tf = posting.tf
        norm = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
        return idf * tf * (self.k1 + 1.0) / norm

    def score_array(self, term: str, index: Index) -> tuple[Int32Array, Float64Array]:
        doc_ids, tfs = index.posting_arrays(term)
        df = int(doc_ids.size)
        if df == 0:
            return doc_ids, np.empty(0, dtype=np.float64)
        n = index.num_docs
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        avgdl = index.avg_doc_length or 1.0
        dl = index.doc_lengths_array[doc_ids].astype(np.float64)
        tf_f = tfs.astype(np.float64)

        scores = np.zeros(df, dtype=np.float64)
        pos = tf_f > 0.0
        if np.any(pos):
            tf_pos = tf_f[pos]
            norm = tf_pos + self.k1 * (1.0 - self.b + self.b * dl[pos] / avgdl)
            scores[pos] = idf * tf_pos * (self.k1 + 1.0) / norm
        return doc_ids, scores

    def __repr__(self) -> str:
        return f"BM25(k1={self.k1}, b={self.b})"


def make_scorer(name: ScorerName) -> Scorer:
    return BM25() if name == "bm25" else TfIdf()
