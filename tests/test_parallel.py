"""Лаба 5: serial, threads і processes дають ОДНАКОВИЙ індекс."""

import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import list_document_paths
from findex.index import build_index, build_partial, make_pipeline, merge
from findex.model import Index, save_index
from findex.parallel import ExecutorName, build_index_parallel, split_paths

VOCAB = [f"слово{i}" for i in range(30)] + ["python", "async", "rust", "м'ясо"]


@pytest.fixture(scope="session")
def many_docs_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """23 невеликі документи: достатньо, щоб ділити на шматки."""
    root = tmp_path_factory.mktemp("many")
    rng = random.Random(7)
    for i in range(23):
        words = [rng.choice(VOCAB) for _ in range(rng.randint(5, 60))]
        (root / f"doc{i:02d}.txt").write_text(" ".join(words), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def many_paths(many_docs_dir: Path) -> list[Path]:
    return list_document_paths(many_docs_dir)


def json_bytes(index: Index, tmp_path: Path, name: str) -> bytes:
    path = tmp_path / f"{name}.json"
    save_index(index, path, "json")
    return path.read_bytes()


# --- M1: build_partial + merge ---------------------------------------------------
@pytest.mark.parametrize("positions", [False, True])
def test_one_partial_equals_lab4(
    many_docs_dir: Path, many_paths: list[Path], positions: bool
) -> None:
    """Послідовний варіант merge([build_partial(усі шляхи)]) == індекс із лаби 4."""
    lab4 = build_index(make_pipeline(many_docs_dir), positions=positions)
    assert merge([build_partial(many_paths, 0, positions)]) == lab4


@pytest.mark.parametrize(("n_paths", "parts"), [(23, 1), (23, 4), (23, 8), (5, 8), (1, 3), (0, 4)])
def test_split_paths_covers_everything_once(n_paths: int, parts: int) -> None:
    paths = [Path(f"{i}.txt") for i in range(n_paths)]
    chunks = split_paths(paths, parts)

    assert [p for chunk in chunks for p in chunk.paths] == paths  # без пропусків і повторів
    sizes = [len(chunk.paths) for chunk in chunks]
    assert max(sizes) - min(sizes) <= 1
    position = 0
    for chunk in chunks:  # діапазони doc_id не перетинаються і йдуть підряд
        assert chunk.first_doc_id == position
        position += len(chunk.paths)


def test_merge_does_not_depend_on_partial_order(many_paths: list[Path]) -> None:
    plan = split_paths(many_paths, 4)
    partials = [build_partial(c.paths, c.first_doc_id) for c in plan]
    assert merge(partials) == merge(reversed(partials))


def test_doc_ids_are_unique_across_chunks(many_paths: list[Path]) -> None:
    plan = split_paths(many_paths, 5)
    all_ids = [i for c in plan for i in build_partial(c.paths, c.first_doc_id).doc_lengths]
    assert sorted(all_ids) == list(range(len(many_paths)))


# --- M2: виконавці -----------------------------------------------------------------
@pytest.mark.parametrize("chunks", [1, 2, 3, 7, 100])
def test_serial_with_many_chunks_equals_lab4(
    many_docs_dir: Path, many_paths: list[Path], chunks: int
) -> None:
    lab4 = build_index(make_pipeline(many_docs_dir), positions=True)
    report = build_index_parallel(many_paths, executor="serial", positions=True, chunks=chunks)
    assert report.built_index == lab4


def test_threads_equal_lab4(many_docs_dir: Path, many_paths: list[Path]) -> None:
    lab4 = build_index(make_pipeline(many_docs_dir), positions=True)
    report = build_index_parallel(many_paths, workers=3, executor="threads", positions=True)
    assert report.built_index == lab4


@pytest.mark.slow
def test_processes_equal_lab4(many_docs_dir: Path, many_paths: list[Path]) -> None:
    lab4 = build_index(make_pipeline(many_docs_dir), positions=True)
    report = build_index_parallel(many_paths, workers=2, executor="processes", positions=True)
    assert report.built_index == lab4


@pytest.mark.slow
def test_all_executors_give_byte_identical_json(many_paths: list[Path], tmp_path: Path) -> None:
    outputs: dict[str, bytes] = {}
    for name, workers in (("serial", 1), ("threads", 3), ("processes", 2)):
        report = build_index_parallel(
            many_paths,
            workers=workers,
            executor=name,
            positions=True,  # type: ignore[arg-type]
        )
        outputs[name] = json_bytes(report.built_index, tmp_path, name)
    assert outputs["serial"] == outputs["threads"] == outputs["processes"]


def test_report_has_sane_numbers(many_paths: list[Path]) -> None:
    report = build_index_parallel(many_paths, workers=2, executor="threads")
    assert report.chunks == 2
    assert report.wall_seconds > 0
    assert 0 <= report.merge_seconds <= report.wall_seconds
    assert report.cpu_seconds >= 0


def test_progress_callback_counts_all_documents(many_paths: list[Path]) -> None:
    done: list[int] = []
    build_index_parallel(many_paths, workers=3, executor="threads", progress=done.append)
    assert sum(done) == len(many_paths)


@pytest.mark.parametrize("kwargs", [{"workers": 0}, {"chunks": 0}])
def test_invalid_arguments(many_paths: list[Path], kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        build_index_parallel(many_paths, **kwargs)


# --- помилка воркера: збірка падає гучно, а не зависає ---------------------------------
def paths_with_bad_file(many_paths: list[Path]) -> list[Path]:
    # NUL у шляху -> ValueError (не OSError!), тож build_partial його не «проковтне»
    return [*many_paths, Path("bad\0name.txt")]


@pytest.mark.parametrize("executor", ["serial", "threads"])
def test_worker_error_propagates(many_paths: list[Path], executor: ExecutorName) -> None:
    with pytest.raises(ValueError, match="null"):
        build_index_parallel(paths_with_bad_file(many_paths), workers=2, executor=executor)


@pytest.mark.slow
def test_process_worker_error_propagates(many_paths: list[Path]) -> None:
    with pytest.raises(ValueError, match="null"):
        build_index_parallel(paths_with_bad_file(many_paths), workers=2, executor="processes")


# --- CLI ----------------------------------------------------------------------------------
def test_cli_threads_equals_serial(runner: CliRunner, many_docs_dir: Path, tmp_path: Path) -> None:
    serial, threaded = tmp_path / "serial.json", tmp_path / "threads.json"
    base = ["index", str(many_docs_dir), "--format", "json", "--positions"]
    assert runner.invoke(app, [*base, "--out", str(serial)]).exit_code == 0
    result = runner.invoke(
        app, [*base, "--out", str(threaded), "--executor", "threads", "--workers", "3"]
    )
    assert result.exit_code == 0, result.output
    assert serial.read_bytes() == threaded.read_bytes()


@pytest.mark.slow
def test_cli_processes_equals_serial(
    runner: CliRunner, many_docs_dir: Path, tmp_path: Path
) -> None:
    serial, parallel = tmp_path / "serial.json", tmp_path / "processes.json"
    base = ["index", str(many_docs_dir), "--format", "json"]
    assert runner.invoke(app, [*base, "--out", str(serial)]).exit_code == 0
    result = runner.invoke(
        app, [*base, "--out", str(parallel), "--executor", "processes", "--workers", "2"]
    )
    assert result.exit_code == 0, result.output
    assert serial.read_bytes() == parallel.read_bytes()


@pytest.mark.parametrize("extra", [["--executor", "magic"], ["--workers", "0"]])
def test_cli_rejects_bad_options(
    runner: CliRunner, many_docs_dir: Path, tmp_path: Path, extra: list[str]
) -> None:
    result = runner.invoke(app, ["index", str(many_docs_dir), "--out", str(tmp_path / "x"), *extra])
    assert result.exit_code == 2  # помилка використання, а не збій програми
