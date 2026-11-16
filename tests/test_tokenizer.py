import pytest

from findex.tokenize import tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, World!", ["hello", "world"]),
        ("", []),
        ("   \n\t ", []),
        ("Привіт, СВІТ!", ["привіт", "світ"]),
        ("м'ясо", ["м'ясо"]),
        ("ЇЖАК Ґанок Єдність", ["їжак", "ґанок", "єдність"]),
        ("a1b2", ["a", "b"]),  # цифри — розділювачі
        ("don't stop", ["don't", "stop"]),
        ("Straße", ["strasse"]),  # casefold, а не lower
    ],
)
def test_tokenize_cases(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


def test_tokenize_normalizes_to_nfc() -> None:
    decomposed = "\u0438\u0306"  # «и» + комбінований «˘»
    assert list(tokenize(decomposed)) == ["\u0439"]  # «й»


def test_tokenize_is_lazy() -> None:
    it = tokenize("one two")
    assert next(it) == "one"
    assert next(it) == "two"
    with pytest.raises(StopIteration):
        next(it)
