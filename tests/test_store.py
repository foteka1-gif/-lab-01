from pathlib import Path

import pytest

from findex.model import Index, IndexFormat, load_index, save_index


@pytest.mark.parametrize("fmt", ["pickle", "json"])
def test_roundtrip_with_positions(index: Index, tmp_path: Path, fmt: IndexFormat) -> None:
    path = tmp_path / f"i.{fmt}"
    save_index(index, path, fmt)
    assert load_index(path, fmt) == index


@pytest.mark.parametrize("fmt", ["pickle", "json"])
def test_roundtrip_without_positions(index_nopos: Index, tmp_path: Path, fmt: IndexFormat) -> None:
    path = tmp_path / f"i.{fmt}"
    save_index(index_nopos, path, fmt)
    loaded = load_index(path, fmt)
    assert loaded == index_nopos
    assert loaded.has_positions is False


def test_json_keeps_unicode_readable(index: Index, tmp_path: Path) -> None:
    path = tmp_path / "i.json"
    save_index(index, path, "json")
    assert "привіт" in path.read_text(encoding="utf-8")


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "nope.bin")
