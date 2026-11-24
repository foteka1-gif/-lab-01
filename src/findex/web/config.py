"""Конфігурація вебсервера зі змінних оточення."""

import os
from dataclasses import dataclass
from pathlib import Path

from ..model import IndexFormat


@dataclass(frozen=True, slots=True)
class Settings:
    index_path: Path | None = None
    index_format: IndexFormat = "pickle"
    semantic_path: Path | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    @classmethod
    def from_env(cls) -> "Settings":
        raw_path = os.getenv("INDEX_PATH", "").strip()
        index_path = Path(raw_path) if raw_path else None

        raw_sem = os.getenv("SEMANTIC_PATH", "").strip()
        if raw_sem:
            semantic_path: Path | None = Path(raw_sem)
        elif index_path is not None:
            candidate = index_path.with_suffix(".npy")
            semantic_path = candidate if candidate.is_file() else None
        else:
            semantic_path = None

        raw_fmt = os.getenv("INDEX_FORMAT", "pickle").strip().lower()
        index_format: IndexFormat = "json" if raw_fmt == "json" else "pickle"

        host = os.getenv("HOST", "0.0.0.0").strip() or "0.0.0.0"
        port = int(os.getenv("PORT", "8000"))
        workers = max(1, int(os.getenv("WORKERS", "1")))
        return cls(
            index_path=index_path,
            index_format=index_format,
            semantic_path=semantic_path,
            host=host,
            port=port,
            workers=workers,
        )
