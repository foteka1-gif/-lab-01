import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: int
    path: str
    text: str


def iter_documents(root: str | Path) -> Iterator[Document]:
    root = Path(root)
    doc_id = 0
    for path in sorted(root.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            log.warning("Не вдалося прочитати %s: %s", path, error)
            continue
        yield Document(doc_id=doc_id, path=str(path), text=text)
        doc_id += 1
