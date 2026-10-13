from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int

@dataclass(frozen=True, slots=True)
class DocMeta:
    path: str
    title: str

