from pathlib import Path

import pytest
from helpers import build_from_tokens

from findex import Posting
from findex.corpus import iter_documents
from findex.index import build_index, make_pipeline
from findex.model import Index, open_index, save_index


def test_counts(index: Index) -> None:
    assert index.num_docs == 5
    assert len(index) == 12
    assert repr(index) == "Index(terms=12, docs=5)"


def test_df_and_doc_length(index: Index) -> None:
    assert index.df("python") == 4
    assert index.df("rust") == 1
    assert index.df("zzz") == 0
    assert index.doc_length(0) == 4
    assert index.doc_length(3) == 41


def test_avg_doc_length(index: Index) -> None:
    assert index.avg_doc_length == pytest.approx(56 / 5)


def test_doc_ids_universe(index: Index) -> None:
    assert index.doc_ids == (0, 1, 2, 3, 4)


def test_mapping_behaviour(index: Index) -> None:
    assert "python" in index
    assert "zzz" not in index
    assert index.get("zzz") is None
    assert index["python"][0].doc_id == 0
    with pytest.raises(KeyError):
        index["zzz"]


def test_postings_are_sorted(index: Index) -> None:
    for plist in index.values():
        doc_ids = [p.doc_id for p in plist]
        assert doc_ids == sorted(doc_ids)


def test_positions_recorded(index: Index) -> None:
    assert index["python"][0] == Posting(doc_id=0, tf=2, positions=(0, 3))
    assert index.has_positions is True


def test_no_positions_by_default(index_nopos: Index) -> None:
    assert index_nopos.has_positions is False
    assert index_nopos["python"][0].positions == ()
    assert index_nopos["python"][0].tf == 2


def test_meta(index: Index) -> None:
    assert index.meta(0).title == "a_alpha"
    assert index.meta(0).path.endswith("a_alpha.txt")


def test_equality_checks_all_parts() -> None:
    a = build_from_tokens([["x", "y"]])
    b = build_from_tokens([["x", "y"]])
    c = build_from_tokens([["x", "y"], ["z"]])
    assert a == b
    assert a != c


def test_iter_documents_skips_non_txt(corpus_dir: Path) -> None:
    docs = list(iter_documents(corpus_dir))
    assert [d.doc_id for d in docs] == [0, 1, 2, 3, 4]
    assert all(d.path.endswith(".txt") for d in docs)


def test_make_pipeline_limit(corpus_dir: Path) -> None:
    assert len(list(make_pipeline(corpus_dir, limit=2))) == 2


def test_build_index_from_empty_pipeline() -> None:
    empty = build_index(iter(()))
    assert len(empty) == 0
    assert empty.avg_doc_length == 0.0


def test_open_index_closes_on_exit(index: Index, tmp_path: Path) -> None:
    path = tmp_path / "i.bin"
    save_index(index, path)
    with open_index(path) as ix:
        assert len(ix) == 12
        assert not ix.closed
    assert ix.closed
    assert len(ix) == 0


def test_open_index_closes_on_exception(index: Index, tmp_path: Path) -> None:
    path = tmp_path / "i.bin"
    save_index(index, path)
    with pytest.raises(RuntimeError), open_index(path) as ix:
        raise RuntimeError("boom")
    assert ix.closed
