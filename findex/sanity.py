"""M2: три sanity-check рейтингу на ВАШОМУ корпусі.
 
python -m findex.sanity index.bin --rare ТЕРМІН_З_df=2 --common ЧАСТИЙ_ТЕРМІН
(--term для перевірки довжини: будь-який термін, що є щонайменше у двох документах)
"""
import argparse
 
from . import Posting
from .model import open_index
from .scoring import BM25, TfIdf
 
 
def check_rare_vs_common(index, scorer, rare, common):
    """1) Рідкий термін (tf=1) важить більше за частий (tf=1)."""
    s_rare = scorer(rare, Posting(doc_id=index[rare][0].doc_id, tf=1), index)
    s_common = scorer(common, Posting(doc_id=index[common][0].doc_id, tf=1), index)
    print(f"[1] {rare!r} (df={index.df(rare)}): {s_rare:.4f}  vs  "
          f"{common!r} (df={index.df(common)}): {s_common:.4f}")
    return s_rare > s_common
 
 
def check_saturation(index, scorer, term):
    """2) BM25: 20-те повторення майже нічого не додає."""
    doc_id = index[term][0].doc_id
    score = {tf: scorer(term, Posting(doc_id=doc_id, tf=tf), index) for tf in (1, 2, 10, 19, 20)}
    first_gain = score[2] - score[1]
    last_gain = score[20] - score[19]
    print(f"[2] tf=1: {score[1]:.4f}, tf=2: {score[2]:.4f}, tf=19: {score[19]:.4f}, "
          f"tf=20: {score[20]:.4f}")
    print(f"    приріст 1->2: {first_gain:.4f}, приріст 19->20: {last_gain:.6f}")
    return last_gain < first_gain / 20
 
 
def check_length_norm(index, scorer, term):
    """3) Короткий документ з одним входженням вище дуже довгого."""
    plist = sorted(index[term], key=lambda p: index.doc_length(p.doc_id))
    short, long_ = plist[0], plist[-1]
    s_short = scorer(term, Posting(doc_id=short.doc_id, tf=1), index)
    s_long = scorer(term, Posting(doc_id=long_.doc_id, tf=1), index)
    print(f"[3] tf=1 у коротшому ({index.doc_length(short.doc_id)} ток.): {s_short:.4f}  vs  "
          f"у довшому ({index.doc_length(long_.doc_id)} ток.): {s_long:.4f}")
    return s_short > s_long
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_file")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle")
    parser.add_argument("--rare", required=True)
    parser.add_argument("--common", required=True)
    parser.add_argument("--term", help="для перевірок 2 і 3 (за замовчуванням --common)")
    args = parser.parse_args()
 
    with open_index(args.index_file, args.format) as index:
        term = args.term or args.common
        for scorer in (TfIdf(), BM25()):
            print(f"\n=== {scorer} ===")
            ok1 = check_rare_vs_common(index, scorer, args.rare, args.common)
            print("    OK" if ok1 else "    FAIL")
            if isinstance(scorer, BM25):
                ok2 = check_saturation(index, scorer, term)
                print("    OK" if ok2 else "    FAIL")
                ok3 = check_length_norm(index, scorer, term)
                print("    OK" if ok3 else "    FAIL")
 
 
if __name__ == "__main__":
    main()