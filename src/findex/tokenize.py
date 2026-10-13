import re
import unicodedata
from typing import Iterator

WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯіІїЇєЄґҐ']+")

def tokenize(text: str) -> Iterator[str]:
    normalized = unicodedata.normalize("NFC", text)
    lowered = normalized.casefold()
    for match in WORD_RE.finditer(lowered):
        yield match.group(0)
