"""Lazy (генератори) проти eager (списки): той самий підрахунок токенів."""

from collections import Counter
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

from .corpus import Document, iter_documents
from .tokenize import tokenize


def run_lazy_pipeline(data_path: Path, limit: int | None = None) -> Counter[str]:
    docs: Iterable[Document] = iter_documents(data_path)
    if limit is not None:
        docs = islice(docs, limit)
    return Counter(tok for doc in docs for tok in tokenize(doc.text))


def run_eager_pipeline(data_path: Path, limit: int | None = None) -> Counter[str]:
    docs = list(iter_documents(data_path))
    if limit is not None:
        docs = docs[:limit]
    all_tokens: list[str] = []
    for doc in docs:
        all_tokens.extend(tokenize(doc.text))
    return Counter(all_tokens)
