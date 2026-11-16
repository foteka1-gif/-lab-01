import pytest

from findex.model import Index
from findex.query import And, Not, Or, Phrase, Term, parse

A, B, C = Term("a"), Term("b"), Term("c")


@pytest.mark.parametrize(
    ("query", "tree"),
    [
        ("a", A),
        ("a b", And(A, B)),  # неявний AND
        ("a AND b", And(A, B)),
        ("a OR b AND c", Or(A, And(B, C))),  # AND сильніший за OR
        ("(a OR b) c", And(Or(A, B), C)),
        ("NOT a", Not(A)),
        ("NOT NOT a", Not(Not(A))),
        ("a b c", And(And(A, B), C)),  # ліва асоціативність
        ('"a b"', Phrase(("a", "b"))),
        ('"a"', A),  # фраза з одного слова — просто терм
        ("a or b", And(And(A, Term("or")), B)),  # малі літери — не оператор
        ("A", A),  # токенізатор опускає регістр
    ],
)
def test_parse_builds_expected_tree(query: str, tree: object) -> None:
    assert parse(query) == tree  # порівнюємо ДЕРЕВА, а не рядки


@pytest.mark.parametrize("query", ["", "   ", '"abc', "(a", "a)", "AND", "a OR"])
def test_parse_errors(query: str) -> None:
    with pytest.raises(ValueError):
        parse(query)


def test_operator_overloading() -> None:
    assert (A & B) == And(A, B)
    assert (A | B) == Or(A, B)
    assert ~A == Not(A)


def test_terms_skip_negated() -> None:
    assert parse("a NOT b OR c").terms() == ["a", "c"]
    assert parse('"x y" z').terms() == ["x", "y", "z"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("python", {0, 1, 2, 3}),
        ("python AND async", {0, 2}),
        ("python NOT async", {1, 3}),
        ("NOT python", {4}),
        ("rust OR привіт", {1, 4}),
        ("zzz", set()),
        ('"async await"', {0}),
        ('"python async"', {0}),  # у gamma слова не поруч
    ],
)
def test_evaluate(index: Index, query: str, expected: set[int]) -> None:
    assert parse(query).evaluate(index) == expected


def test_phrase_requires_positions(index_nopos: Index) -> None:
    with pytest.raises(ValueError, match="--positions"):
        parse('"async await"').evaluate(index_nopos)
