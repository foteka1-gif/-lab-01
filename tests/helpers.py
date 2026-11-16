"""Спільні функції для тестів (їх можна імпортувати і з Hypothesis-тестів)."""

from collections.abc import Sequence

from findex.index import build_index
from findex.model import Index


def build_from_tokens(docs: Sequence[Sequence[str]], positions: bool = False) -> Index:
    """Індекс прямо з токенів, без файлів: doc0, doc1, ..."""
    pipeline = ((i, f"doc{i}.txt", f"doc{i}", tokens) for i, tokens in enumerate(docs))
    return build_index(pipeline, positions=positions)
