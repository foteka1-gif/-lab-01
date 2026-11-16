"""Побудова інвертованого індексу з потоку документів."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path

from . import DocMeta, Posting
from .corpus import Document, iter_documents
from .decorators import timed
from .model import Index
from .tokenize import tokenize

# (doc_id, path, title, tokens)
type PipelineItem = tuple[int, str, str, Iterable[str]]


@timed
def build_index(corpus_pipeline: Iterable[PipelineItem], positions: bool = False) -> Index:
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, path, title, tokens in corpus_pipeline:
        doc_meta[doc_id] = DocMeta(path=str(path), title=title)

        term_positions: dict[str, list[int]] = {}
        if positions:
            for pos, token in enumerate(tokens):
                term_positions.setdefault(token, []).append(pos)
            term_counts = Counter({term: len(plist) for term, plist in term_positions.items()})
        else:
            term_counts = Counter(tokens)

        doc_lengths[doc_id] = sum(term_counts.values())

        for term, tf in term_counts.items():
            postings[term].append(
                Posting(
                    doc_id=doc_id,
                    tf=tf,
                    positions=tuple(term_positions[term]) if positions else (),
                )
            )

    return Index(dict(postings), doc_lengths, doc_meta)


def make_pipeline(root: str | Path, limit: int | None = None) -> Iterator[PipelineItem]:
    docs: Iterable[Document] = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)
    for doc in docs:
        title = Path(doc.path).stem
        yield doc.doc_id, doc.path, title, tokenize(doc.text)