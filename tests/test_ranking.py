import unittest

from findex import Posting
from findex.scoring import BM25, TfIdf
from findex.search import search, make_matcher, SearchResult
from findex.snippets import make_snippet
from findex.evaluation import precision_at_k
from findex.decorators import timed
from tests.common import make_index


class RankingTests(unittest.TestCase):
    def setUp(self):
        filler = " ".join(f"w{i}" for i in range(200))
        docs = {f"doc{i}": "the common text" for i in range(20)}
        docs["rare_doc"] = "the common text zebra"
        docs["short_doc"] = "needle"
        docs["long_doc"] = "needle " + filler
        self.ix = make_index(docs)

    def test_rare_term_beats_common(self):
        for scorer in (TfIdf(), BM25()):
            rare = scorer("zebra", Posting(0, 1), self.ix)
            common = scorer("common", Posting(0, 1), self.ix)
            self.assertGreater(rare, common)

    def test_bm25_saturation(self):
        s = lambda tf: BM25()("needle", Posting(0, tf), self.ix)
        self.assertLess(s(20) - s(19), (s(2) - s(1)) / 20)

    def test_bm25_short_beats_long(self):
        results = search(self.ix, "needle", BM25())
        self.assertEqual([r.title for r in results[:2]], ["short_doc", "long_doc"])

    def test_scorers_are_interchangeable(self):
        for scorer in (TfIdf(), BM25(), BM25(k1=1.2, b=0.5)):
            self.assertTrue(search(self.ix, "zebra", scorer))

    def test_results_sortable(self):
        a, b = SearchResult(1.0, 1, "x"), SearchResult(2.0, 2, "y")
        self.assertEqual(sorted([b, a]), [a, b])

    def test_top_k(self):
        self.assertEqual(len(search(self.ix, "common", k=5)), 5)


class CacheAndDecoratorTests(unittest.TestCase):
    def test_wraps_keeps_name_and_doc(self):
        @timed
        def f():
            """doc"""
        self.assertEqual((f.__name__, f.__doc__), ("f", "doc"))
        self.assertEqual(search.__name__, "search")

    def test_matcher_cache_hit(self):
        ix = make_index({"a": "x y", "b": "x"})
        matcher = make_matcher(ix)
        with self.assertLogs("findex", level="DEBUG") as cm:
            matcher("x"); matcher("x")
        self.assertIn("MISS", cm.output[0])
        self.assertIn("HIT", cm.output[1])
        self.assertEqual(matcher.cache_info().hits, 1)


class SnippetAndEvalTests(unittest.TestCase):
    def test_snippet_highlights_and_trims(self):
        text = "a " * 100 + "Python and asyncio live here " + "b " * 100
        snip = make_snippet(text, ["python", "asyncio"])
        self.assertIn("**Python**", snip)
        self.assertIn("**asyncio**", snip)
        self.assertTrue(snip.startswith("…") and snip.endswith("…"))
        self.assertLess(len(snip), 220)

    def test_precision_at_5(self):
        results = [SearchResult(1, i, t) for i, t in enumerate("abcde")]
        self.assertEqual(precision_at_k(results, {"a", "c"}, 5), 0.4)


if __name__ == "__main__":
    unittest.main()
