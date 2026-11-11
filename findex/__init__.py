
from dataclasses import dataclass
 
 
@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    # ЛАБ 3: позиції токенів у документі (потрібні для фраз і сніпетів).
    # За замовчуванням порожній кортеж -> код з лаб 1-2 працює без змін.
    positions: tuple[int, ...] = ()
 
 
@dataclass(frozen=True, slots=True)
class DocMeta:
    path: str
    title: str