import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex import cli
from findex.cli import app


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    # Без цього rich переносить довгі рядки таблиць по 80 колонках, і підрядки «розвалюються».
    monkeypatch.setattr(cli.console, "width", 200)


def test_help(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("index", "search", "stats"):
        assert command in result.stdout


def test_index_creates_file(runner: CliRunner, corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "sub" / "idx.bin"
    result = runner.invoke(app, ["index", str(corpus_dir), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Індекс збережено" in result.stdout


def test_index_limit(runner: CliRunner, corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "idx.bin"
    result = runner.invoke(app, ["index", str(corpus_dir), "--out", str(out), "--limit", "2"])
    assert result.exit_code == 0, result.output
    assert "docs=2" in result.stdout


def test_index_missing_corpus(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "Помилка" in result.stderr
    assert "Traceback" not in result.output


def test_index_empty_corpus(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", str(tmp_path)])
    assert result.exit_code == 1
    assert "немає .txt" in result.stderr


def test_verbose_sends_logs_to_stderr(runner: CliRunner, corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "idx.bin"
    result = runner.invoke(app, ["-v", "index", str(corpus_dir), "--out", str(out)])
    assert result.exit_code == 0
    assert "Папка документів" in result.stderr
    assert "Папка документів" not in result.stdout


def test_search_prints_table(runner: CliRunner, index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "rust"])
    assert result.exit_code == 0, result.output
    assert "b_beta" in result.stdout


def test_search_no_results(runner: CliRunner, index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "zzz"])
    assert result.exit_code == 0
    assert "Нічого не знайдено" in result.stdout


def test_search_tfidf_option(runner: CliRunner, index_file: Path) -> None:
    result = runner.invoke(
        app, ["search", str(index_file), "python", "--scorer", "tfidf", "-k", "2"]
    )
    assert result.exit_code == 0, result.output


def test_search_json_pipes_cleanly(runner: CliRunner, index_file: Path) -> None:
    # навіть з -vv у stdout лише JSON: логи йдуть у stderr
    result = runner.invoke(app, ["-vv", "search", str(index_file), "python", "--json", "-k", "3"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 3
    assert rows[0]["rank"] == 1
    assert set(rows[0]) == {"rank", "score", "doc_id", "title", "snippet"}
    assert result.stderr  # діагностика справді там


def test_search_json_keeps_unicode(runner: CliRunner, index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "привіт", "--json"])
    assert result.exit_code == 0, result.output
    assert "**Привіт**" in result.stdout  # кирилиця, а не \u041f...
    assert json.loads(result.stdout.splitlines()[0])["title"] == "e_epsilon"


def test_search_missing_index(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", str(tmp_path / "nope.bin"), "python"])
    assert result.exit_code == 1
    assert "не знайдено" in result.stderr
    assert "Traceback" not in result.output


def test_search_corrupt_index(runner: CliRunner, tmp_path: Path) -> None:
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"this is not an index")
    result = runner.invoke(app, ["search", str(bad), "python"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output


def test_search_bad_query_is_one_line(runner: CliRunner, index_file: Path) -> None:
    result = runner.invoke(app, ["search", str(index_file), "(python"])
    assert result.exit_code == 1
    assert len(result.stderr.strip().splitlines()) == 1
    assert "Помилка" in result.stderr


def test_phrase_without_positions(runner: CliRunner, corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "nopos.bin"
    assert runner.invoke(app, ["index", str(corpus_dir), "--out", str(out)]).exit_code == 0
    result = runner.invoke(app, ["search", str(out), '"async await"'])
    assert result.exit_code == 1
    assert "--positions" in result.stderr


def test_stats(runner: CliRunner, index_file: Path) -> None:
    result = runner.invoke(app, ["stats", str(index_file)])
    assert result.exit_code == 0, result.output
    assert "Документів" in result.stdout
    assert "python (4)" in result.stdout
