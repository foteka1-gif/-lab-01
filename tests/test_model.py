import unittest

from findex.model import Index, open_index, save_index
from tests.common import make_index
import tempfile, os


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.ix = make_index({"a": "python python async", "b": "java python", "c": "go"})

    def test_container_protocol(self):
        self.assertEqual(len(self.ix), 4)
        self.assertIn("python", self.ix)
        self.assertNotIn("rust", self.ix)
        self.assertEqual(sorted(self.ix), ["async", "go", "java", "python"])
        self.assertEqual(self.ix["python"][0].tf, 2)
        with self.assertRaises(KeyError):
            self.ix["rust"]

    def test_repr_and_attrs(self):
        self.assertEqual(repr(self.ix), "Index(terms=4, docs=3)")
        self.assertEqual(self.ix.num_docs, 3)
        self.assertEqual(self.ix.df("python"), 2)
        self.assertEqual(self.ix.df("rust"), 0)
        self.assertAlmostEqual(self.ix.avg_doc_length, 6 / 3)
        self.assertTrue(self.ix.has_positions)

    def test_avg_doc_length_is_cached(self):
        _ = self.ix.avg_doc_length
        self.assertIn("avg_doc_length", self.ix.__dict__)

    def test_context_manager_closes_even_on_exception(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ix.bin")
            save_index(self.ix, path)
            holder = {}
            with self.assertRaises(RuntimeError):
                with open_index(path) as ix:
                    holder["ix"] = ix
                    self.assertEqual(len(ix), 4)
                    raise RuntimeError("boom")
            self.assertTrue(holder["ix"].closed)
            self.assertEqual(len(holder["ix"]), 0)

    def test_json_roundtrip_keeps_positions(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ix.json")
            save_index(self.ix, path, "json")
            with open_index(path, "json") as ix:
                self.assertEqual(ix["python"], self.ix["python"])
                self.assertTrue(ix.has_positions)


if __name__ == "__main__":
    unittest.main()
