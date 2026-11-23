"""Кеш правил robots.txt для кожного хоста."""

import asyncio
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

from .fetch import DEFAULT_USER_AGENT


class RobotsCache:
    """Асинхронно завантажує та кешує robots.txt окремо для кожного (scheme, netloc)."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._timeout = timeout_seconds
        self._parsers: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _origin_key(url: str) -> str:
        parts = urlsplit(url)
        return f"{parts.scheme.lower()}://{parts.netloc.lower()}"

    @staticmethod
    def _robots_url(url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), "/robots.txt", "", ""))

    async def _load_parser(self, url: str) -> RobotFileParser:
        robots_url = self._robots_url(url)
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            async with asyncio.timeout(self._timeout):
                response = await self._client.get(robots_url, follow_redirects=True)
            if 200 <= response.status_code < 300:
                parser.parse(response.text.splitlines())
            elif response.status_code in (401, 403):
                parser.parse(["User-agent: *", "Disallow: /"])
            else:
                parser.parse([])
        except (TimeoutError, httpx.HTTPError):
            parser.parse([])
        return parser

    async def can_fetch(self, url: str) -> bool:
        """Повертає True, якщо обхід url дозволено правилами robots.txt хоста."""
        key = self._origin_key(url)
        cached = self._parsers.get(key)
        if cached is not None:
            return cached.can_fetch(self._user_agent, url)

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self._parsers.get(key)
            if cached is None:
                cached = await self._load_parser(url)
                self._parsers[key] = cached
        return cached.can_fetch(self._user_agent, url)

    def clear(self) -> None:
        self._parsers.clear()
        self._locks.clear()
