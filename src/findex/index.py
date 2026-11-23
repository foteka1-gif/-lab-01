"""Побудова інвертованого індексу.

Два шляхи, що дають ОДНАКОВИЙ результат:
  * build_index   — з потоку готових документів (лаба 4);
  * build_partial + merge — по шматках шляхів (лаба 5): шматки можна віддати потокам/процесам.
"""

import logging
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from itertools import islice
from pathlib import Path
from typing import NamedTuple

from . import DocMeta, Posting
from .corpus import Document, iter_documents
from .decorators import timed
from .model import Index
from .tokenize import tokenize

log = logging.getLogger(__name__)

# (doc_id, path, title, tokens)
type PipelineItem = tuple[int, str, str, Iterable[str]]


class PartialIndex(NamedTuple):
    """Індекс частини корпусу. Компактний і picklable: його повертає воркер."""

    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]


def _add_document(
    postings: dict[str, list[Posting]],
    doc_lengths: dict[int, int],
    doc_meta: dict[int, DocMeta],
    doc_id: int,
    path: str,
    title: str,
    tokens: Iterable[str],
    positions: bool,
) -> None:
    """Додає один документ до накопичувачів. Спільна для обох шляхів побудови."""
    doc_meta[doc_id] = DocMeta(path=path, title=title)

    term_positions: dict[str, list[int]] = {}
    if positions:
        for pos, token in enumerate(tokens):
            term_positions.setdefault(token, []).append(pos)
        term_counts = Counter({term: len(plist) for term, plist in term_positions.items()})
    else:
        term_counts = Counter(tokens)

    doc_lengths[doc_id] = sum(term_counts.values())

    for term, tf in term_counts.items():
        postings.setdefault(term, []).append(
            Posting(
                doc_id=doc_id,
                tf=tf,
                positions=tuple(term_positions[term]) if positions else (),
            )
        )


@timed
def build_index(corpus_pipeline: Iterable[PipelineItem], positions: bool = False) -> Index:
    """Лаба 4: послідовна побудова з потоку документів."""
    postings: dict[str, list[Posting]] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, path, title, tokens in corpus_pipeline:
        _add_document(postings, doc_lengths, doc_meta, doc_id, path, title, tokens, positions)

    return Index(postings, doc_lengths, doc_meta)


def make_pipeline(root: str | Path, limit: int | None = None) -> Iterator[PipelineItem]:
    docs: Iterable[Document] = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)
    for doc in docs:
        title = doc.title or Path(doc.path).stem
        yield doc.doc_id, doc.path, title, tokenize(doc.text)


# --- лаба 5: map (build_partial) + reduce (merge) -----------------------------
def build_partial(
    doc_paths: Sequence[Path],
    first_doc_id: int = 0,
    positions: bool = False,
) -> PartialIndex:
    """Будує індекс частини корпусу. ВОРКЕР САМ ЧИТАЄ ФАЙЛИ: у нього летять шляхи, а не тексти.

    Функція рівня модуля (не вкладена і не lambda), тому її можна віддати іншому процесові.
    doc_id документа = first_doc_id + його позиція в doc_paths, тож діапазони шматків
    не перетинаються. Файл, який не вдалося прочитати, пропускається (його doc_id лишається
    вільним).
    """
    postings: dict[str, list[Posting]] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for offset, path in enumerate(doc_paths):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            log.warning("Не вдалося прочитати %s: %s", path, error)
            continue
        _add_document(
            postings,
            doc_lengths,
            doc_meta,
            first_doc_id + offset,
            str(path),
            path.stem,
            tokenize(text),
            positions,
        )

    return PartialIndex(postings, doc_lengths, doc_meta)


@timed
def merge(partials: Iterable[PartialIndex]) -> Index:
    """Зливає частини в один Index. Порядок частин не важливий: результат завжди той самий.

    Це послідовна частина алгоритму, тому її час окремо визначає стелю прискорення
    (закон Амдала). Терміни й постінги сортуються, щоб serial/threads/processes давали
    ідентичні (навіть побайтово у JSON) індекси.
    """
    postings: dict[str, list[Posting]] = {}
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for part in partials:
        for term, plist in part.postings.items():
            postings.setdefault(term, []).extend(plist)
        doc_lengths.update(part.doc_lengths)
        doc_meta.update(part.doc_meta)

    for plist in postings.values():
        plist.sort(key=lambda posting: posting.doc_id)  # шматки можуть прийти в будь-якому порядку

    return Index(
        dict(sorted(postings.items())),
        dict(sorted(doc_lengths.items())),
        dict(sorted(doc_meta.items())),
    )
