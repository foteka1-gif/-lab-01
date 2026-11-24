"""Витяг заголовка, видимого тексту та посилань з HTML."""

from dataclasses import dataclass
from html.parser import HTMLParser


@dataclass(frozen=True, slots=True)
class ParsedPage:
    title: str
    text: str
    links: tuple[str, ...]


_IGNORED_TAGS = {"script", "style", "noscript", "template"}
_BLOCK_TAGS = {
    "p",
    "div",
    "br",
    "li",
    "ul",
    "ol",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "tr",
    "td",
    "th",
    "section",
    "article",
    "header",
    "footer",
    "blockquote",
    "pre",
}


class _PageExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignore_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self._links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower()
        if lower_tag in _IGNORED_TAGS:
            self._ignore_depth += 1
            return
        if self._ignore_depth > 0:
            return
        if lower_tag == "title":
            self._in_title = True
        elif lower_tag == "a":
            for attr_name, attr_val in attrs:
                if attr_name.lower() == "href" and attr_val:
                    self._links.append(attr_val.strip())
        elif lower_tag in _BLOCK_TAGS:
            self._text_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag in _IGNORED_TAGS and self._ignore_depth > 0:
            self._ignore_depth -= 1
            return
        if lower_tag == "title":
            self._in_title = False
        elif lower_tag in _BLOCK_TAGS and self._ignore_depth == 0:
            self._text_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._ignore_depth > 0:
            return
        if self._in_title:
            self._title_parts.append(data)
        self._text_parts.append(data)

    def build(self) -> ParsedPage:
        title = " ".join("".join(self._title_parts).split())
        text = " ".join("".join(self._text_parts).split())
        return ParsedPage(title=title, text=text, links=tuple(self._links))


def parse_html(html: str) -> ParsedPage:
    """Розбирає HTML-рядок і повертає ParsedPage(title, text, links)."""
    extractor = _PageExtractor()
    extractor.feed(html)
    extractor.close()
    return extractor.build()
