import logging

import pytest

from findex.decorators import timed


def test_timed_preserves_metadata() -> None:
    @timed
    def f() -> int:
        """doc"""
        return 1

    assert f.__name__ == "f"
    assert f.__doc__ == "doc"
    assert f() == 1


def test_timed_logs_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    @timed
    def work() -> None:
        pass

    with caplog.at_level(logging.DEBUG, logger="findex.decorators"):
        work()
    assert "work took" in caplog.text


def test_timed_logs_even_on_exception(caplog: pytest.LogCaptureFixture) -> None:
    @timed
    def boom() -> None:
        raise RuntimeError

    with caplog.at_level(logging.DEBUG, logger="findex.decorators"), pytest.raises(RuntimeError):
        boom()
    assert "boom took" in caplog.text
