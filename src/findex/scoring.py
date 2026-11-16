"""Scorer-протокол + TF-IDF + BM25. Duck typing: наслідування не потрібне."""

import math
from typing import Literal, Protocol

from . import Posting
from .model import Index

type ScorerName = Literal["bm25", "tfidf"]


class Scorer(Protocol):
    def __call__(self, term: str, posting: Posting, index: Index) -> float: ...


class TfIdf:
    """(1 + ln tf) * ln(N / df). Терміни, що є в усіх документах, мають вагу 0."""

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf <= 0:
            return 0.0
        idf = math.log(index.num_docs / df)
        return (1 + math.log(posting.tf)) * idf

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
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))  # варіант Lucene, завжди >= 0
        avgdl = index.avg_doc_length or 1.0
        dl = index.doc_length(posting.doc_id)
        tf = posting.tf
        norm = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
        return idf * tf * (self.k1 + 1) / norm

    def __repr__(self) -> str:
        return f"BM25(k1={self.k1}, b={self.b})"


def make_scorer(name: ScorerName) -> Scorer:
    return BM25() if name == "bm25" else TfIdf()