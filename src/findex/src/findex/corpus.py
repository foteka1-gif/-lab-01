import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: int
    path: str
    text: str
    title: str = ""


def list_document_paths(root: str | Path) -> list[Path]:
    """Усі .txt у папці, відсортовані. Позиція шляху в цьому списку і є його doc_id."""
    return sorted(Path(root).rglob("*.txt"))


def write_jsonl_record(
    stream: IO[str],
    *,
    url: str,
    title: str,
    text: str,
    status: int = 200,
) -> None:
    """Записує одну сторінку у потік JSONL (по одному JSON-об'єкту на рядок)."""
    record: dict[str, str | int] = {
        "url": url,
        "title": title,
        "text": text,
        "status": status,
    }
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def _iter_jsonl_documents(jsonl_path: Path) -> Iterator[Document]:
    doc_id = 0
    try:
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    raw_obj: object = json.loads(line)
                except json.JSONDecodeError as error:
                    log.warning("Пропущено битий рядок %d у %s: %s", line_no, jsonl_path, error)
                    continue
                if not isinstance(raw_obj, dict):
                    continue
                data = cast(dict[str, object], raw_obj)
                url = str(data.get("url", f"{jsonl_path}:{line_no}"))
                title = str(data.get("title", ""))
                text = str(data.get("text", ""))
                yield Document(doc_id=doc_id, path=url, text=text, title=title)
                doc_id += 1
    except OSError as error:
        log.warning("Не вдалося прочитати %s: %s", jsonl_path, error)


def iter_documents(root: str | Path) -> Iterator[Document]:
    """Читає документи з папки (.txt) або з .jsonl-файлу (результату краулера)."""
    target = Path(root)
    if target.is_file():
        yield from _iter_jsonl_documents(target)
        return

    doc_id = 0
    for path in list_document_paths(target):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            log.warning("Не вдалося прочитати %s: %s", path, error)
            continue
        yield Document(doc_id=doc_id, path=str(path), text=text, title=path.stem)
        doc_id += 1
