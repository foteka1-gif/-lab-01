"""Властивості (Hypothesis): генерує сотні входів і шукає контрприклад."""

import tempfile
from pathlib import Path

import pytest
from helpers import build_from_tokens
from hypothesis import given, settings
from hypothesis import strategies as st

from findex import Posting
from findex.boolean import merge_and, merge_or, set_operation
from findex.model import IndexFormat, load_index, save_index
from findex.query import And, Node, Not, Or, Phrase, Term, parse
from findex.tokenize import tokenize

ALPHABET = "abcdefghijklmnopqrstuvwxyzабвгдеєжзиіїйклмнопрстуфхцчшщьюя'"
words = st.text(alphabet=ALPHABET, min_size=1, max_size=8)
docs_strategy = st.lists(st.lists(words, max_size=12), min_size=1, max_size=6)


# 1. Round-trip: load(save(index)) == index
@pytest.mark.parametrize("fmt", ["pickle", "json"])
@settings(deadline=None)  # файловий I/O може бути повільнішим за 200 мс
@given(docs=docs_strategy, positions=st.booleans())
def test_save_load_roundtrip(fmt: IndexFormat, docs: list[list[str]], positions: bool) -> None:
    index = build_from_tokens(docs, positions)
    # TemporaryDirectory, а не фікстура tmp_path: вона одна на всі приклади Hypothesis
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.bin"
        save_index(index, path, fmt)
        assert load_index(path, fmt) == index


# 2. Інваріант: постінги відсортовані, без дублів, tf > 0
@given(docs_strategy)
def test_postings_sorted_unique_positive(docs: list[list[str]]) -> None:
    index = build_from_tokens(docs)
    for plist in index.values():
        doc_ids = [p.doc_id for p in plist]
        assert doc_ids == sorted(set(doc_ids))
        assert all(p.tf > 0 for p in plist)


# 3. Оракул: швидкий merge збігається з очевидно-правильними множинами
sorted_ids = st.lists(st.integers(0, 200), unique=True).map(sorted)


@given(a=sorted_ids, b=sorted_ids)
def test_merge_matches_set_oracle(a: list[int], b: list[int]) -> None:
    pa = [Posting(i, 1) for i in a]
    pb = [Posting(i, 1) for i in b]
    assert [p.doc_id for p in merge_and(pa, pb)] == sorted(set(a) & set(b))
    assert [p.doc_id for p in merge_or(pa, pb)] == sorted(set(a) | set(b))
    assert merge_and(pa, pb) == set_operation(pa, pb, "AND")
    assert merge_or(pa, pb) == set_operation(pa, pb, "OR")


# 4. Ідемпотентність токенізатора і відсутність порожніх токенів
@given(st.text())
def test_tokenize_idempotent(text: str) -> None:
    tokens = list(tokenize(text))
    assert all(t and not any(c.isspace() for c in t) for t in tokens)
    assert list(tokenize(" ".join(tokens))) == tokens


# 5. Парсер: render() -> parse() повертає рівне дерево
leaf = st.sampled_from(["python", "async", "rust", "go"]).map(Term) | st.lists(
    st.sampled_from(["async", "await", "python"]), min_size=2, max_size=3
).map(lambda ws: Phrase(tuple(ws)))

trees = st.recursive(
    leaf,
    lambda children: (
        st.builds(And, children, children)
        | st.builds(Or, children, children)
        | st.builds(Not, children)
    ),
    max_leaves=6,
)


@given(trees)
def test_render_parse_roundtrip(node: Node) -> None:
    assert parse(node.render()) == node
