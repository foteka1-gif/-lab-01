"""M1: Index як об'єкт, що поводиться «як Python»."""
import functools
from collections.abc import Mapping
from contextlib import contextmanager
 
from .decorators import timed
from .store import save_pickle, load_pickle, save_json, load_json
 
 
class Index(Mapping):
    """Інвертований індекс: term -> список Posting.
 
    Наслідуємо collections.abc.Mapping: реалізуємо лише __getitem__, __len__,
    __iter__ — а keys(), items(), values(), get(), == отримуємо безкоштовно.
    """
 
    def __init__(self, postings, doc_lengths, doc_meta):
        self._postings = postings
        self._doc_lengths = doc_lengths
        self._doc_meta = doc_meta
        self._closed = False
 
    # --- протокол контейнера ---------------------------------------------
    def __len__(self):                 # len(index)  -> кількість термінів
        return len(self._postings)
 
    def __iter__(self):                # for term in index
        return iter(self._postings)
 
    def __getitem__(self, term):       # index["python"]; KeyError, якщо немає
        return self._postings[term]
 
    def __contains__(self, term):      # "python" in index (швидше, ніж у Mapping)
        return term in self._postings
 
    def __repr__(self):
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"
 
    # --- обчислювані атрибути (без get_*) --------------------------------
    @property
    def num_docs(self) -> int:
        return len(self._doc_lengths)
 
    @functools.cached_property
    def avg_doc_length(self) -> float:  # рахуємо один раз і кешуємо в instance
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / self.num_docs
 
    @functools.cached_property
    def doc_ids(self) -> tuple:         # «всесвіт» документів для NOT
        return tuple(sorted(self._doc_lengths))
 
    @functools.cached_property
    def has_positions(self) -> bool:    # чи були позиції (--positions)
        return any(plist[0].positions for plist in self._postings.values() if plist)
 
    # --- методи ------------------------------------------------------------
    def doc_length(self, doc_id) -> int:
        return self._doc_lengths[doc_id]
 
    def df(self, term) -> int:          # document frequency
        return len(self._postings.get(term, ()))
 
    def meta(self, doc_id):
        return self._doc_meta[doc_id]
 
    def to_parts(self):
        """Те саме, що повертав build_index у лаб 2: (postings, doc_lengths, doc_meta)."""
        return self._postings, self._doc_lengths, self._doc_meta
 
    @property
    def closed(self) -> bool:
        return self._closed
 
    def close(self):
        """Звільняє пам'ять. Викликається з open_index навіть при винятку."""
        self._postings, self._doc_lengths, self._doc_meta = {}, {}, {}
        for name in ("avg_doc_length", "doc_ids", "has_positions"):
            self.__dict__.pop(name, None)  # скидаємо cached_property
        self._closed = True
 
 
_LOADERS = {"pickle": load_pickle, "json": load_json}
_SAVERS = {"pickle": save_pickle, "json": save_json}
 
 
@timed
def load_index(path, fmt="pickle") -> Index:
    return Index(*_LOADERS[fmt](path))
 
 
@timed
def save_index(index, path, fmt="pickle"):
    _SAVERS[fmt](index.to_parts(), path)
 
 
@contextmanager
def open_index(path, fmt="pickle"):
    """with open_index(p) as ix: ...
    Все до yield — це __enter__, усе після — __exit__ (finally спрацює й при raise/return)."""
    index = load_index(path, fmt)
    try:
        yield index
    finally:
        index.close()