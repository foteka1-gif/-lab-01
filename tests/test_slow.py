import random

import pytest
from helpers import build_from_tokens

from findex.scoring import BM25
from findex.search import search


@pytest.mark.slow
def test_larger_corpus_search_is_consistent() -> None:
    rng = random.Random(42)
    vocab = [f"word{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(300)]
    docs = [[rng.choice(vocab) for _ in range(rng.randint(5, 80))] for _ in range(3000)]
    index = build_from_tokens(docs, positions=True)

    results = search(index, f"{vocab[1]} OR {vocab[2]} OR {vocab[3]}", BM25(), k=20)

    assert len(results) == 20

