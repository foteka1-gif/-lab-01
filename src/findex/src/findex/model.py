"""Index як об'єкт, що поводиться «як Python»."""

import functools
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from typing import Literal

from . import DocMeta, Posting
from .decorators import timed
from .store import (
    IndexParts,
    PathLike,
    load_json,
    load_pickle,
    save_json,
    save_pickle,
)

type IndexFormat = Literal["pickle", "json"]


class Index(Mapping[str, list[Posting]]):
    """Інвертований індекс: term -> список Posting.

    Наслідуємо Mapping: реалізуємо лише __getitem__, __len__, __iter__ —
    а keys(), items(), values(), get() отримуємо безкоштовно.
    """

    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
    ) -> None:
        self._postings = postings
        self._doc_lengths = doc_lengths
        self._doc_meta = doc_meta
        self._closed = False

    # --- протокол контейнера ---------------------------------------------
    def __len__(self) -> int:
        return len(self._postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __getitem__(self, key: str) -> list[Posting]:
        return self._postings[key]

    def __contains__(self, key: object) -> bool:
        return key in self._postings

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    def __eq__(self, other: object) -> bool:
        # Mapping.__eq__ порівнював би лише postings; нам потрібні й довжини, і метадані.
        if not isinstance(other, Index):
            return NotImplemented
        return self.to_parts() == other.to_parts()

    # --- обчислювані атрибути ---------------------------------------------
    @property
    def num_docs(self) -> int:
        return len(self._doc_lengths)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / self.num_docs

    @functools.cached_property
    def doc_ids(self) -> tuple[int, ...]:  # «всесвіт» документів для NOT
        return tuple(sorted(self._doc_lengths))

    @functools.cached_property
    def has_positions(self) -> bool:
        return any(bool(plist[0].positions) for plist in self._postings.values() if plist)

    # --- методи ------------------------------------------------------------
    def doc_length(self, doc_id: int) -> int:
        return self._doc_lengths[doc_id]

    def df(self, term: str) -> int:
        """Document frequency."""
        return len(self._postings.get(term, ()))

    def meta(self, doc_id: int) -> DocMeta:
        return self._doc_meta[doc_id]

    def to_parts(self) -> IndexParts:
        return self._postings, self._doc_lengths, self._doc_meta

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Звільняє пам'ять. open_index викликає це навіть при винятку."""
        self._postings, self._doc_lengths, self._doc_meta = {}, {}, {}
        for name in ("avg_doc_length", "doc_ids", "has_positions"):
            self.__dict__.pop(name, None)  # скидаємо cached_property
        self._closed = True


_LOADERS: dict[IndexFormat, Callable[[PathLike], IndexParts]] = {
    "pickle": load_pickle,
    "json": load_json,
}
_SAVERS: dict[IndexFormat, Callable[[IndexParts, PathLike], None]] = {
    "pickle": save_pickle,
    "json": save_json,
}


@timed
def load_index(path: PathLike, fmt: IndexFormat = "pickle") -> Index:
    postings, doc_lengths, doc_meta = _LOADERS[fmt](path)
    return Index(postings, doc_lengths, doc_meta)


@timed
def save_index(index: Index, path: PathLike, fmt: IndexFormat = "pickle") -> None:
    _SAVERS[fmt](index.to_parts(), path)


@contextmanager
def open_index(path: PathLike, fmt: IndexFormat = "pickle") -> Generator[Index]:
    """with open_index(p) as ix: ...  Усе після yield виконається і при винятку."""
    index = load_index(path, fmt)
    try:
        yield index
    finally:
        index.close()
