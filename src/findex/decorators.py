import functools
import logging
import time
from collections.abc import Callable

log = logging.getLogger(__name__)


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """@timed над def search == search = timed(search). Пише час у DEBUG-лог."""

    @functools.wraps(fn)  # без цього __name__ == "wrapper", а docstring губиться
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            log.debug("%s took %.3f ms", fn.__name__, elapsed * 1000)

    return wrapper