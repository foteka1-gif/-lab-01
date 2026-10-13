import logging
from pathlib import Path
from typing import Iterator, NamedTuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Document(NamedTuple):
    doc_id: str
    path: Path
    text: str

def iter_documents(root: Path) -> Iterator[Document]:
    if not root.exists():
        raise FileNotFoundError(f"Директорію {root} не знайдено.")
    for path in root.rglob("*.txt"):
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                yield Document(doc_id=path.stem, path=path, text=text)
            except Exception as e:
                logger.error(f"Помилка при зчитуванні файлу {path}: {e}")
                continue
