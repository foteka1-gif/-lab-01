"""Асинхронне завантаження сторінок із таймаутами, повторами (backoff + jitter) та Retry-After."""

import asyncio
import email.utils
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC

import httpx

DEFAULT_USER_AGENT = "findex-crawler/0.6 (+https://github.com; contact=student@example.edu)"


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    status: int
    content_type: str
    text: str
    num_bytes: int
    elapsed: float
    attempts: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and self.error is None


def parse_retry_after(header_value: str | None, now: float | None = None) -> float | None:
    """Розбирає заголовок Retry-After (секунди або HTTP-дата) у затримку в секундах."""
    if not header_value:
        return None
    cleaned = header_value.strip()
    if not cleaned:
        return None
    try:
        seconds = float(cleaned)
        return max(0.0, seconds)
    except ValueError:
        pass
    try:
        parsed_dt = email.utils.parsedate_to_datetime(cleaned)
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=UTC)
        current_ts = time.time() if now is None else now
        return max(0.0, parsed_dt.timestamp() - current_ts)
    except (TypeError, ValueError):
        return None


def _is_retryable_status(status: int) -> bool:
    return status == 429 or 500 <= status < 600


async def fetch_url(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout_seconds: float = 10.0,
    max_retries: int = 3,
    base_backoff: float = 0.25,
    max_backoff: float = 10.0,
    rng: Callable[[], float] = random.random,
) -> FetchResult:
    """Завантажує URL з повторами лише для 429, 5xx та таймаутів (до max_retries разів)."""
    started = time.perf_counter()
    last_status = 0
    last_error: str | None = None

    for attempt in range(max_retries + 1):
        retry_delay: float | None = None
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await client.get(url, follow_redirects=True)
            last_status = response.status_code
            content_type = response.headers.get("content-type", "")
            raw_bytes = response.content
            num_bytes = len(raw_bytes)
            final_url = str(response.url)

            if not _is_retryable_status(last_status):
                text = response.text if "text" in content_type or not content_type else ""
                err = None if 200 <= last_status < 300 else f"HTTP {last_status}"
                return FetchResult(
                    url=url,
                    final_url=final_url,
                    status=last_status,
                    content_type=content_type,
                    text=text,
                    num_bytes=num_bytes,
                    elapsed=time.perf_counter() - started,
                    attempts=attempt + 1,
                    error=err,
                )

            last_error = f"HTTP {last_status}"
            retry_delay = parse_retry_after(response.headers.get("retry-after"))

        except TimeoutError:
            last_status = 0
            last_error = "timeout"
        except httpx.TooManyRedirects:
            return FetchResult(
                url=url,
                final_url=url,
                status=0,
                content_type="",
                text="",
                num_bytes=0,
                elapsed=time.perf_counter() - started,
                attempts=attempt + 1,
                error="too_many_redirects",
            )
        except httpx.HTTPError as exc:
            last_status = 0
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt >= max_retries:
            break

        if retry_delay is None:
            exp_delay = min(max_backoff, base_backoff * (2**attempt))
            retry_delay = exp_delay * (0.5 + rng())
        else:
            retry_delay = min(max_backoff, retry_delay)

        await asyncio.sleep(retry_delay)

    return FetchResult(
        url=url,
        final_url=url,
        status=last_status,
        content_type="",
        text="",
        num_bytes=0,
        elapsed=time.perf_counter() - started,
        attempts=max_retries + 1,
        error=last_error or "unknown_error",
    )
