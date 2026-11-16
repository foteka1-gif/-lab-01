import json
import pickle
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from . import DocMeta, Posting

type PathLike = str | Path
type IndexParts = tuple[dict[str, list[Posting]], dict[int, int], dict[int, DocMeta]]


class PostingJson(TypedDict):
    doc_id: int
    tf: int
    positions: NotRequired[list[int]]


class DocMetaJson(TypedDict):
    path: str
    title: str


class IndexJson(TypedDict):
    postings: dict[str, list[PostingJson]]
    doc_lengths: dict[str, int]
    doc_meta: dict[str, DocMetaJson]


def save_pickle(data: IndexParts, path: PathLike) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: PathLike) -> IndexParts:
    # Завантажувати можна тільки довірені pickle-файли.
    with open(path, "rb") as f:
        # pickle.load повертає Any; форму гарантує save_pickle.
        return cast(IndexParts, pickle.load(f))


def _posting_to_json(p: Posting) -> PostingJson:
    item: PostingJson = {"doc_id": p.doc_id, "tf": p.tf}
    if p.positions:  # позиції пишемо, лише якщо вони є
        item["positions"] = list(p.positions)
    return item


def save_json(index_data: IndexParts, path: PathLike) -> None:
    postings, doc_lengths, doc_meta = index_data
    full_data: IndexJson = {
        "postings": {
            term: [_posting_to_json(p) for p in posting_list]
            for term, posting_list in postings.items()
        },
        "doc_lengths": {str(d): n for d, n in doc_lengths.items()},
        "doc_meta": {str(d): {"path": m.path, "title": m.title} for d, m in doc_meta.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)


def load_json(path: PathLike) -> IndexParts:
    with open(path, encoding="utf-8") as f:
        # json.load повертає Any; форму гарантує save_json.
        full_data = cast(IndexJson, json.load(f))

    postings = {
        term: [
            Posting(doc_id=p["doc_id"], tf=p["tf"], positions=tuple(p.get("positions", [])))
            for p in raw_list
        ]
        for term, raw_list in full_data["postings"].items()
    }
    doc_lengths = {int(k): v for k, v in full_data["doc_lengths"].items()}
    doc_meta = {
        int(k): DocMeta(path=v["path"], title=v["title"]) for k, v in full_data["doc_meta"].items()
    }
    return postings, doc_lengths, doc_meta