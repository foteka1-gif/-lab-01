import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.index import build_index, make_pipeline
from findex.model import Index

# Імена підібрані так, щоб алфавітний порядок = doc_id: a_alpha=0, b_beta=1, ...
CORPUS = {
    "a_alpha": "python async await python",
    "b_beta": "python rust memory safety",
    "c_gamma": "python go concurrency async",
    "d_delta": "Python " + "filler " * 40,  # довгий документ (41 токен)
    "e_epsilon": "Привіт, світ! М'ясо",
}
# df: python=4, async=2, rust=1; довжини: 4, 4, 4, 41, 3 -> avgdl = 56 / 5


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    for name, text in CORPUS.items():
        (root / f"{name}.txt").write_text(text, encoding="utf-8")
    (root / "ignored.md").write_text("python", encoding="utf-8")  # не .txt -> ігнорується
    return root


@pytest.fixture(scope="session")
def index(corpus_dir: Path) -> Index:
    """Індекс З позиціями. Session-scope: тести його НЕ змінюють."""
    return build_index(make_pipeline(corpus_dir), positions=True)


@pytest.fixture(scope="session")
def index_nopos(corpus_dir: Path) -> Index:
    return build_index(make_pipeline(corpus_dir), positions=False)


@pytest.fixture(scope="session")
def runner() -> CliRunner:
    try:
        return CliRunner(mix_stderr=False)  # type: ignore[call-arg]  # click < 8.2
    except TypeError:
        return CliRunner()  # click >= 8.2: stdout і stderr розділені завжди


@pytest.fixture(scope="session")
def index_file(
    tmp_path_factory: pytest.TempPathFactory, corpus_dir: Path, runner: CliRunner
) -> Path:
    path = tmp_path_factory.mktemp("idx") / "index.bin"
    result = runner.invoke(app, ["index", str(corpus_dir), "--out", str(path), "--positions"])
    assert result.exit_code == 0, result.output
    return path


@pytest.fixture(autouse=True)
def _reset_findex_logging() -> object:
    """CLI вішає StreamHandler на stderr CliRunner'а; після тесту цей потік закритий."""
    yield
    package_logger = logging.getLogger("findex")
    for handler in list(package_logger.handlers):
        package_logger.removeHandler(handler)
    package_logger.setLevel(logging.NOTSET)
