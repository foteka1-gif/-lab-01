
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


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
            text = path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            yield Document(
                doc_id=doc_id,
                path=str(path),
                text=text
            )

            doc_id += 1

        except OSError as error:
            print(f"Не вдалося прочитати {path}: {error}")
