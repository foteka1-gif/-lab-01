import pytest

from findex import Posting
from findex.boolean import merge_and, merge_not, merge_or, set_operation


def make(*ids: int) -> list[Posting]:
    return [Posting(i, 1) for i in ids]


def ids(postings: list[Posting]) -> list[int]:
    return [p.doc_id for p in postings]


def test_merge_and_sums_tf() -> None:
    result = merge_and(make(1, 2, 3), make(2, 3, 4))
    assert ids(result) == [2, 3]
    assert all(p.tf == 2 for p in result)


def test_merge_and_with_empty() -> None:
    assert merge_and([], make(1, 2)) == []


def test_merge_or() -> None:
    result = merge_or(make(1, 2, 3), make(2, 3, 4))
    assert ids(result) == [1, 2, 3, 4]
    assert [p.tf for p in result] == [1, 2, 2, 1]


def test_merge_not_zeroes_tf() -> None:
    result = merge_not(make(1, 2), make(1, 2, 3))
    assert result == [Posting(3, 0)]


@pytest.mark.parametrize(
    ("operation", "expected"),
    [("AND", [2]), ("OR", [1, 2, 3]), ("NOT", [1])],
)
def test_set_operation(operation: str, expected: list[int]) -> None:
    result = set_operation(make(1, 2), make(2, 3), operation)  # type: ignore[arg-type]
    assert ids(result) == expected


@pytest.mark.parametrize("operation", ["AND", "OR"])
def test_merge_agrees_with_set_operation(operation: str) -> None:
    a, b = make(1, 3, 5, 7), make(3, 4, 5, 8)
    merge = merge_and if operation == "AND" else merge_or
    assert merge(a, b) == set_operation(a, b, operation)  # type: ignore[arg-type]
