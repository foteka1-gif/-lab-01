import unittest

from findex.query import parse, Term, Phrase, And, Or, Not
from tests.common import make_index


class ParserTreeTests(unittest.TestCase):
    def test_precedence_or_below_and(self):
        self.assertEqual(parse("a OR b c"), Or(Term("a"), And(Term("b"), Term("c"))))

    def test_explicit_and(self):
        self.assertEqual(parse("a AND b"), And(Term("a"), Term("b")))

    def test_not_and_parentheses(self):
        self.assertEqual(
            parse("python AND (async OR await) NOT java"),
            And(And(Term("python"), Or(Term("async"), Term("await"))), Not(Term("java"))),
        )

    def test_phrase(self):
        self.assertEqual(
            parse('python "event loop"'),
            And(Term("python"), Phrase(("event", "loop"))),
        )

    def test_full_example_from_assignment(self):
        tree = parse('python AND (async OR await) NOT java "event loop"')
        expected = (Term("python") & (Term("async") | Term("await"))
                    & ~Term("java") & Phrase(("event", "loop")))
        self.assertEqual(tree, expected)

    def test_operator_overloads_build_nodes(self):
        self.assertEqual(Term("a") & Term("b"), And(Term("a"), Term("b")))
        self.assertEqual(Term("a") | Term("b"), Or(Term("a"), Term("b")))
        self.assertEqual(~Term("a"), Not(Term("a")))

    def test_lowercase_or_is_a_word(self):
        self.assertEqual(parse("a or b"), And(And(Term("a"), Term("or")), Term("b")))

    def test_errors(self):
        for bad in ["", "(a", "a)", 'a "b', "AND", "a AND"]:
            with self.subTest(query=bad), self.assertRaises(ValueError):
                parse(bad)


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.ix = make_index({
            "d0": "python event loop async",
            "d1": "python loop event",
            "d2": "java event loop",
            "d3": "python await",
        })

    def ids(self, q):
        return parse(q).evaluate(self.ix)

    def test_boolean(self):
        self.assertEqual(self.ids("python"), {0, 1, 3})
        self.assertEqual(self.ids("python AND (async OR await)"), {0, 3})
        self.assertEqual(self.ids("python NOT async"), {1, 3})
        self.assertEqual(self.ids("NOT python"), {2})

    def test_phrase_uses_positions(self):
        self.assertEqual(self.ids('"event loop"'), {0, 2})  # d1 має "loop event"

    def test_phrase_without_positions_raises(self):
        ix = make_index({"a": "event loop"}, positions=False)
        with self.assertRaises(ValueError):
            parse('"event loop"').evaluate(ix)


if __name__ == "__main__":
    unittest.main()
