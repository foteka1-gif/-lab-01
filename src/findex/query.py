"""Мова запитів. Рекурсивний спуск -> дерево вузлів Term/Phrase/And/Or/Not.

Граматика (пріоритет: NOT > AND > OR; між термінами неявний AND):
    or_expr  -> and_expr ('OR' and_expr)*
    and_expr -> not_expr (['AND'] not_expr)*
    not_expr -> 'NOT' not_expr | atom
    atom     -> TERM | PHRASE | '(' or_expr ')'
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, NamedTuple

from .model import Index
from .tokenize import tokenize


class Node(ABC):
    """Базовий вузол: перевантажує &, |, ~ -> запити можна складати в коді."""

    @abstractmethod
    def evaluate(self, index: Index) -> set[int]: ...

    @abstractmethod
    def terms(self) -> list[str]:
        """Позитивні терміни (без тих, що під NOT) — для ранжування й підсвітки."""

    @abstractmethod
    def render(self) -> str:
        """Рядок запиту, який parse() перетворить на рівне дерево."""

    def __and__(self, other: "Node") -> "Node":
        return And(self, other)

    def __or__(self, other: "Node") -> "Node":
        return Or(self, other)

    def __invert__(self) -> "Node":
        return Not(self)


@dataclass(frozen=True)
class Term(Node):
    text: str

    def evaluate(self, index: Index) -> set[int]:
        return {p.doc_id for p in index.get(self.text, [])}

    def terms(self) -> list[str]:
        return [self.text]

    def render(self) -> str:
        return self.text


@dataclass(frozen=True)
class Phrase(Node):
    words: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not index.has_positions:
            raise ValueError("Для фраз потрібні позиції: побудуйте індекс з прапорцем --positions.")
        per_word: list[dict[int, frozenset[int]]] = [
            {p.doc_id: frozenset(p.positions) for p in index.get(w, [])} for w in self.words
        ]
        candidates = set(per_word[0]).intersection(*per_word[1:])
        result: set[int] = set()
        for doc_id in candidates:
            for start in per_word[0][doc_id]:
                if all(start + i in per_word[i][doc_id] for i in range(1, len(self.words))):
                    result.add(doc_id)
                    break
        return result

    def terms(self) -> list[str]:
        return list(self.words)

    def render(self) -> str:
        return '"' + " ".join(self.words) + '"'


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)

    def terms(self) -> list[str]:
        return self.left.terms() + self.right.terms()

    def render(self) -> str:
        return f"({self.left.render()} AND {self.right.render()})"


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)

    def terms(self) -> list[str]:
        return self.left.terms() + self.right.terms()

    def render(self) -> str:
        return f"({self.left.render()} OR {self.right.render()})"


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def evaluate(self, index: Index) -> set[int]:
        return set(index.doc_ids) - self.child.evaluate(index)

    def terms(self) -> list[str]:
        return []

    def render(self) -> str:
        return f"NOT {self.child.render()}"


# --- лексер ------------------------------------------------------------------
_TOKEN_RE = re.compile(r'"[^"]*"|[()]|[^\s()"]+')
_OPERATORS = {"AND", "OR", "NOT"}  # лише ВЕЛИКИМИ: "or" малими — звичайне слово

type TokenKind = Literal["LP", "RP", "AND", "OR", "NOT", "TERM", "PHRASE"]


class Token(NamedTuple):
    kind: TokenKind
    text: str = ""  # для TERM
    words: tuple[str, ...] = ()  # для PHRASE


def _operator(raw: str) -> TokenKind | None:
    if raw == "AND":
        return "AND"
    if raw == "OR":
        return "OR"
    if raw == "NOT":
        return "NOT"
    return None


def lex(query: str) -> list[Token]:
    if query.count('"') % 2:
        raise ValueError("Непарні лапки у запиті.")
    tokens: list[Token] = []
    raw_tokens: list[str] = _TOKEN_RE.findall(query)
    for raw in raw_tokens:
        operator = _operator(raw)
        if raw == "(":
            tokens.append(Token("LP"))
        elif raw == ")":
            tokens.append(Token("RP"))
        elif operator is not None:
            tokens.append(Token(operator))
        elif raw.startswith('"'):
            words = tuple(tokenize(raw[1:-1]))  # той самий токенізатор, що й в індексі
            if len(words) == 1:
                tokens.append(Token("TERM", text=words[0]))
            elif words:
                tokens.append(Token("PHRASE", words=words))
        else:
            for word in tokenize(raw):
                tokens.append(Token("TERM", text=word))
    return tokens


# --- парсер ------------------------------------------------------------------
class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> TokenKind | None:
        return self.tokens[self.pos].kind if self.pos < len(self.tokens) else None

    def take(self) -> Token:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def or_expr(self) -> Node:
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            node = Or(node, self.and_expr())
        return node

    def and_expr(self) -> Node:
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

    def not_expr(self) -> Node:
        if self.peek() == "NOT":
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Node:
        kind = self.peek()
        if kind == "TERM":
            return Term(self.take().text)
        if kind == "PHRASE":
            return Phrase(self.take().words)
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
