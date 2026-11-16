"""findex: пошуковий рушій з інвертованим індексом."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    # Позиції токенів у документі (потрібні для фраз і сніпетів).
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    path: str
    title: str
