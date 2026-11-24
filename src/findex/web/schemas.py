"""Pydantic-схеми на вході та виході HTTP API (усередині лишаються dataclass-и)."""

from typing import Literal

from pydantic import BaseModel, Field

from ..semantic import SearchMode


class SearchQueryParams(BaseModel):
    q: str = Field(..., min_length=1, max_length=500, description="Пошуковий запит")
    k: int = Field(default=10, ge=1, le=100, description="Кількість результатів на сторінку")
    scorer: Literal["bm25", "tfidf"] = Field(default="bm25", description="Функція ранжування")
    mode: SearchMode = Field(
        default="lexical", description="Режим пошуку: lexical, semantic, hybrid"
    )
    page: int = Field(default=1, ge=1, le=1000, description="Номер сторінки (з 1)")


class SearchHitOut(BaseModel):
    rank: int
    doc_id: int
    title: str
    path: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    scorer: Literal["bm25", "tfidf"]
    mode: SearchMode = "lexical"
    page: int
    per_page: int
    total_hits: int
    total_pages: int
    elapsed_ms: float
    results: list[SearchHitOut]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    length: int
    text: str


class StatsResponse(BaseModel):
    num_docs: int
    num_terms: int
    total_tokens: int
    avg_doc_length: float
    has_positions: bool
    semantic_loaded: bool = False
    top_terms: list[tuple[str, int]]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    index_loaded: bool
    semantic_loaded: bool = False
    num_docs: int = 0
