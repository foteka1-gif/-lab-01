"""M3: мова запитів. Рекурсивний спуск -> дерево вузлів Term/Phrase/And/Or/Not.
 
Граматика (пріоритет: NOT > AND > OR; між термінами неявний AND):
    or_expr  -> and_expr ('OR' and_expr)*
    and_expr -> not_expr (['AND'] not_expr)*
    not_expr -> 'NOT' not_expr | atom
    atom     -> TERM | PHRASE | '(' or_expr ')'
"""
import re
from dataclasses import dataclass
 
from .tokenize import tokenize
 
 
class Node:
    """Базовий вузол: перевантажує &, |, ~ -> запити можна складати в коді."""
 
    def evaluate(self, index) -> set:
        raise NotImplementedError
 
    def terms(self) -> list:
        """Позитивні терміни (без тих, що під NOT) — для ранжування й підсвітки."""
        raise NotImplementedError
 
    def __and__(self, other):
        return And(self, other) if isinstance(other, Node) else NotImplemented
 
    def __or__(self, other):
        return Or(self, other) if isinstance(other, Node) else NotImplemented
 
    def __invert__(self):
        return Not(self)
 
 
@dataclass(frozen=True)
class Term(Node):
    text: str
 
    def evaluate(self, index):
        return {p.doc_id for p in index.get(self.text, ())}
 
    def terms(self):
        return [self.text]
 
 
@dataclass(frozen=True)
class Phrase(Node):
    words: tuple
 
    def evaluate(self, index):
        if not index.has_positions:
            raise ValueError(
                "Для фраз потрібні позиції: побудуйте індекс з прапорцем --positions."
            )
        per_word = [
            {p.doc_id: frozenset(p.positions) for p in index.get(w, ())}
            for w in self.words
        ]
        candidates = set(per_word[0]).intersection(*per_word[1:])
        result = set()
        for doc_id in candidates:
            for start in per_word[0][doc_id]:
                if all(start + i in per_word[i][doc_id] for i in range(1, len(self.words))):
                    result.add(doc_id)
                    break
        return result
 
    def terms(self):
        return list(self.words)
 
 
@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node
 
    def evaluate(self, index):
        return self.left.evaluate(index) & self.right.evaluate(index)
 
    def terms(self):
        return self.left.terms() + self.right.terms()
 
 
@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node
 
    def evaluate(self, index):
        return self.left.evaluate(index) | self.right.evaluate(index)
 
    def terms(self):
        return self.left.terms() + self.right.terms()
 
 
@dataclass(frozen=True)
class Not(Node):
    child: Node
 
    def evaluate(self, index):
        return set(index.doc_ids) - self.child.evaluate(index)
 
    def terms(self):
        return []
 
 
# --- лексер ------------------------------------------------------------------
_TOKEN_RE = re.compile(r'"[^"]*"|[()]|[^\s()"]+')
_OPERATORS = {"AND", "OR", "NOT"}  # лише ВЕЛИКИМИ: "or" малими — звичайне слово
 
 
def lex(query: str) -> list:
    if query.count('"') % 2:
        raise ValueError("Непарні лапки у запиті.")
    tokens = []
    for raw in _TOKEN_RE.findall(query):
        if raw == "(":
            tokens.append(("LP", raw))
        elif raw == ")":
            tokens.append(("RP", raw))
        elif raw in _OPERATORS:
            tokens.append((raw, raw))
        elif raw.startswith('"'):
            words = tuple(tokenize(raw[1:-1]))  # той самий токенізатор, що й в індексі
            if len(words) == 1:
                tokens.append(("TERM", words[0]))
            elif words:
                tokens.append(("PHRASE", words))
        else:
            for word in tokenize(raw):
                tokens.append(("TERM", word))
    return tokens
 
 
# --- парсер ------------------------------------------------------------------
class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
 
    def peek(self):
        return self.tokens[self.pos][0] if self.pos < len(self.tokens) else None
 
    def take(self):
        token = self.tokens[self.pos]
        self.pos += 1
        return token
 
    def or_expr(self):
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            node = Or(node, self.and_expr())
        return node
 
    def and_expr(self):
        node = self.not_expr()
        while True:
            kind = self.peek()
            if kind == "AND":
                self.take()
                node = And(node, self.not_expr())
            elif kind in ("TERM", "PHRASE", "NOT", "LP"):  # неявний AND
                node = And(node, self.not_expr())
            else:
                return node
 
    def not_expr(self):
        if self.peek() == "NOT":
            self.take()
            return Not(self.not_expr())
        return self.atom()
 
    def atom(self):
        kind = self.peek()
        if kind == "TERM":
            return Term(self.take()[1])
        if kind == "PHRASE":
            return Phrase(self.take()[1])
        if kind == "LP":
            self.take()
            node = self.or_expr()
            if self.peek() != "RP":
                raise ValueError("Неправильні дужки.")
            self.take()
            return node
        raise ValueError(f"Несподіваний елемент запиту: {kind or 'кінець запиту'}.")
 
 
def parse(query: str) -> Node:
    tokens = lex(query)
    if not tokens:
        raise ValueError("Порожній запит.")
    parser = _Parser(tokens)
    node = parser.or_expr()
    if parser.pos != len(tokens):
        raise ValueError("Зайві символи у запиті (перевірте дужки).")
    return node