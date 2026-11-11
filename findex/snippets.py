"""M4: сніпет = вікно ±80 символів навколо найкращого входження + підсвітка термів."""
import itertools
import re
from pathlib import Path
 
 
def _plain(text, limit):
    return " ".join(text[:limit].split())
 
 
def make_snippet(text, terms, window=80, mark=("**", "**")):
    terms = [t for t in dict.fromkeys(terms) if t]
    if not terms:
        return _plain(text, 2 * window)
 
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\w'])(?:{alternatives})(?![\w'])", re.IGNORECASE)
 
    matches = [
        (m.start(), m.end(), m.group(0).casefold())
        for m in itertools.islice(pattern.finditer(text), 500)
    ]
    if not matches:
        return _plain(text, 2 * window)
 
    def coverage(match):
        """Скільки РІЗНИХ термінів потрапляє у вікно навколо цього входження."""
        lo, hi = match[0] - window, match[1] + window
        return len({t for a, b, t in matches if a >= lo and b <= hi})
 
    best = max(matches, key=coverage)  # при рівності — перше входження
    lo = max(0, best[0] - window)
    hi = min(len(text), best[1] + window)
 
    piece = pattern.sub(lambda m: mark[0] + m.group(0) + mark[1], text[lo:hi])
    piece = " ".join(piece.split())
    return ("…" if lo > 0 else "") + piece + ("…" if hi < len(text) else "")
 
 
def snippet_for(meta, terms, **kwargs):
    text = Path(meta.path).read_text(encoding="utf-8", errors="replace")
    return make_snippet(text, terms, **kwargs)