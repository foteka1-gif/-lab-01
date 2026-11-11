import functools
import logging
import time
 
log = logging.getLogger("findex")
 
 
def setup_logging(level=logging.DEBUG):
    """Викликається в CLI, щоб сліди @timed і cache HIT/MISS було видно в консолі."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )
 
 
def timed(fn):
    """@timed над def search  ==  search = timed(search)."""
 
    @functools.wraps(fn)  # без цього __name__ == "wrapper", а docstring губиться
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - t0
            log.debug("%s took %.3f ms", fn.__name__, elapsed * 1000)
 
    return wrapper