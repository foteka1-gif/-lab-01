"""Веб-інтерфейс та HTTP API для findex на базі FastAPI."""

from .app import app, create_app

__all__ = ["app", "create_app"]
