"""Index як об'єкт, що поводиться «як Python» + кешовані масиви NumPy int32."""

import functools
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from typing import Literal

import numpy as np
from numpy.typing import NDArray

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
type Int32Array = NDArray[np.int32]


class Index(Mapping[str, list[Posting]]):
    """Інвертований індекс: term -> список Posting + векторне представлення NumPy int32."""

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
        self._np_postings: dict[str, tuple[Int32Array, Int32Array]] = {}

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
    def doc_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._doc_lengths))

    @functools.cached_property
    def max_doc_id(self) -> int:
        return max(self._doc_lengths) if self._doc_lengths else -1

    @functools.cached_property
    def doc_lengths_array(self) -> Int32Array:
        """Щільний масив довжин документів int32, індексований за doc_id."""
        if not self._doc_lengths:
            return np.empty(0, dtype=np.int32)
        arr = np.zeros(self.max_doc_id + 1, dtype=np.int32)
        for doc_id, length in self._doc_lengths.items():
            arr[doc_id] = length
        return arr

    @functools.cached_property
    def valid_docs_mask(self) -> NDArray[np.bool_]:
        """Булева маска існуючих doc_id (для випадків, коли doc_id мають пропуски)."""
        if not self._doc_lengths:
            return np.empty(0, dtype=np.bool_)
        mask = np.zeros(self.max_doc_id + 1, dtype=np.bool_)
        for doc_id in self._doc_lengths:
            mask[doc_id] = True
        return mask

    @functools.cached_property
    def has_positions(self) -> bool:
        return any(bool(plist[0].positions) for plist in self._postings.values() if plist)

    # --- методи ------------------------------------------------------------
    def doc_length(self, doc_id: int) -> int:
        return self._doc_lengths[doc_id]

    def df(self, term: str) -> int:
        return len(self._postings.get(term, ()))

    def meta(self, doc_id: int) -> DocMeta:
        return self._doc_meta[doc_id]

    def posting_arrays(self, term: str) -> tuple[Int32Array, Int32Array]:
        """Повертає (doc_ids: int32, tfs: int32) для терма з кешуванням."""
        cached = self._np_postings.get(term)
        if cached is not None:
            return cached
        plist = self._postings.get(term)
        if not plist:
            empty = np.empty(0, dtype=np.int32)
            pair = (empty, empty)
            self._np_postings[term] = pair
            return pair
        doc_ids = np.fromiter((p.doc_id for p in plist), dtype=np.int32, count=len(plist))
        tfs = np.fromiter((p.tf for p in plist), dtype=np.int32, count=len(plist))
        pair = (doc_ids, tfs)
        self._np_postings[term] = pair
        return pair

    def warm_numpy_cache(self) -> None:
        """Попередньо будує int32-масиви для всіх термів індексу."""
        _ = self.doc_lengths_array
        _ = self.valid_docs_mask
        for term in self._postings:
            self.posting_arrays(term)

    def to_parts(self) -> IndexParts:
        return self._postings, self._doc_lengths, self._doc_meta

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Звільняє пам'ять. open_index викликає це навіть при винятку."""
        self._postings, self._doc_lengths, self._doc_meta = {}, {}, {}
        self._np_postings.clear()
        for name in (
            "avg_doc_length",
            "doc_ids",
            "max_doc_id",
            "doc_lengths_array",
            "valid_docs_mask",
            "has_positions",
        ):
            self.__dict__.pop(name, None)
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
    index = load_index(path, fmt)
    try:
        yield index
    finally:
        index.close()
