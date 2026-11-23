"""Канонізація та нормалізація URL для веб-краулера."""

import posixpath
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

_ALLOWED_SCHEMES = {"http", "https"}
_DEFAULT_PORTS = {"http": 80, "https": 443}


def canonicalize_url(raw_url: str, base_url: str | None = None) -> str | None:
    """Приводить URL до канонічного вигляду або повертає None, якщо схема не http/https.

    Правила:
      * схема й хост у нижньому регістрі;
      * стандартні порти (:80 для http, :443 для https) прибираються;
      * фрагмент після # відкидається;
      * шлях нормалізується (порожній -> '/', зайві крапки згортаються);
      * query-параметри сортуються за ключем і значенням.
    """
    cleaned = raw_url.strip()
    if not cleaned:
        return None

    joined = urljoin(base_url, cleaned) if base_url else cleaned
    parts = urlsplit(joined)

    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES or not parts.hostname:
        return None

    host = parts.hostname.lower()
    try:
        port = parts.port
    except ValueError:
        return None

    if port is not None and port != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{port}"
    else:
        netloc = host

    raw_path = parts.path or "/"
    had_trailing_slash = raw_path.endswith("/") and raw_path != "/"
    normalized_path = posixpath.normpath(raw_path)
    if not normalized_path.startswith("/"):
        normalized_path = "/" + normalized_path
    if had_trailing_slash and not normalized_path.endswith("/"):
        normalized_path += "/"

    sorted_query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))

    return urlunsplit((scheme, netloc, normalized_path, sorted_query, ""))


def same_host(url_a: str, url_b: str) -> bool:
    """Перевіряє, чи належать дві адреси одному хосту (і порту)."""
    a = urlsplit(url_a)
    b = urlsplit(url_b)
    return (a.hostname or "").lower() == (b.hostname or "").lower() and a.port == b.port
