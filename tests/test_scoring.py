import pytest
from helpers import build_from_tokens

from findex import Posting
from findex.model import Index
from findex.scoring import BM25, Scorer, TfIdf, make_scorer


@pytest.mark.parametrize("scorer", [TfIdf(), BM25()], ids=repr)
def test_rare_term_outweighs_common(index: Index, scorer: Scorer) -> None:
    """1) Рідкий термін (tf=1) важить більше за частий (tf=1)."""
    rare = scorer("rust", Posting(index["rust"][0].doc_id, tf=1), index)
    common = scorer("python", Posting(index["python"][0].doc_id, tf=1), index)
    assert rare > common


def test_bm25_saturation(index: Index) -> None:
    """2) BM25: 20-те повторення майже нічого не додає."""
    scorer = BM25()
    doc_id = index["python"][0].doc_id
    score = {tf: scorer("python", Posting(doc_id, tf), index) for tf in (1, 2, 19, 20)}
    first_gain = score[2] - score[1]
    last_gain = score[20] - score[19]
    assert last_gain < first_gain / 20


def test_bm25_length_normalization(index: Index) -> None:
    """3) Короткий документ з одним входженням вище дуже довгого."""
    scorer = BM25()
    plist = sorted(index["python"], key=lambda p: index.doc_length(p.doc_id))
    short, long_ = plist[0], plist[-1]
    assert index.doc_length(short.doc_id) < index.doc_length(long_.doc_id)
    assert scorer("python", Posting(short.doc_id, 1), index) > scorer(
        "python", Posting(long_.doc_id, 1), index
    )


@pytest.mark.parametrize("scorer", [TfIdf(), BM25()], ids=repr)
def test_unknown_term_scores_zero(index: Index, scorer: Scorer) -> None:
    assert scorer("zzz", Posting(0, 1), index) == 0.0


def test_tfidf_zero_when_term_in_every_doc() -> None:
    ix = build_from_tokens([["x", "a"], ["x", "b"]])
    assert TfIdf()("x", ix["x"][0], ix) == 0.0


def test_bm25_never_negative() -> None:
    ix = build_from_tokens([["x", "a"], ["x", "b"]])
    assert BM25()("x", ix["x"][0], ix) >= 0.0


def test_reprs() -> None:
    assert repr(TfIdf()) == "TfIdf()"
    assert repr(BM25()) == "BM25(k1=1.5, b=0.75)"


def test_make_scorer() -> None:
    assert isinstance(make_scorer("bm25"), BM25)
    assert isinstance(make_scorer("tfidf"), TfIdf)
