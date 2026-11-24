"""Семантичний та гібридний пошук (фрагменти -> L2-нормалізована матриця .npy -> RRF)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from .model import Index
from .search import SearchResult

type SearchMode = Literal["lexical", "semantic", "hybrid"]
type Float32Matrix = NDArray[np.float32]

_EMBED_DIM = 384


@dataclass(frozen=True, slots=True)
class ChunkMeta:
    doc_id: int
    chunk_id: int
    text: str


@dataclass(slots=True)
class SemanticIndex:
    """Матриця нормалізованих ембедингів фрагментів (N x D, float32) + їхні метадані."""

    embeddings: Float32Matrix
    chunks: list[ChunkMeta]

    def save(self, base_path: Path) -> tuple[Path, Path]:
        """Зберігає матрицю у <base>.npy та опис фрагментів у <base>.chunks.json."""
        base_path.parent.mkdir(parents=True, exist_ok=True)
        npy_path = base_path.with_suffix(".npy")
        meta_path = base_path.with_suffix(".chunks.json")
        np.save(npy_path, self.embeddings)
        payload = [
            {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text} for c in self.chunks
        ]
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return npy_path, meta_path

    @classmethod
    def load(cls, base_path: Path) -> "SemanticIndex":
        npy_path = base_path.with_suffix(".npy")
        meta_path = base_path.with_suffix(".chunks.json")
        raw_arr = np.load(npy_path)
        embeddings = np.asarray(raw_arr, dtype=np.float32)
        raw_items = cast(
            list[dict[str, object]],
            json.loads(meta_path.read_text(encoding="utf-8")),
        )
        chunks = [
            ChunkMeta(
                doc_id=int(cast(int, item["doc_id"])),
                chunk_id=int(cast(int, item["chunk_id"])),
                text=str(item["text"]),
            )
            for item in raw_items
        ]
        return cls(embeddings=embeddings, chunks=chunks)


def _split_chunks(text: str, chunk_words: int = 80, overlap: int = 20) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = max(1, chunk_words - overlap)
    out: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_words]
        if piece:
            out.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return out


def _fnv1a_hash(token: str) -> int:
    h = 2166136261
    for ch in token.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def encode_texts(texts: list[str], dim: int = _EMBED_DIM) -> Float32Matrix:
    """Кодує тексти у L2-нормалізовані вектори float32 (символьні 3-4-грами + основи слів)."""
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)

    mat = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        words = cleaned.split()
        if not words:
            continue

        for w in words:
            stem = w[:5] if len(w) >= 5 else w
            for feat, weight in ((w, 1.5), (stem, 1.2)):
                h = _fnv1a_hash(feat)
                col = h % dim
                sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
                mat[row_idx, col] += sign * weight

            padded = f"<{w}>"
            for n in (3, 4):
                if len(padded) >= n:
                    for i in range(len(padded) - n + 1):
                        gram = padded[i : i + n]
                        h = _fnv1a_hash(gram)
                        col = h % dim
                        sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
                        mat[row_idx, col] += sign * 0.65

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    non_zero = norms > 1e-8
    mat = np.where(non_zero, mat / np.maximum(norms, 1e-8), 0.0).astype(np.float32)
    return mat


def build_semantic_index(
    index: Index,
    chunk_words: int = 80,
    overlap: int = 20,
) -> SemanticIndex:
    """Ділить документи корпусу на фрагменти, рахує ембединги й нормалізує вектори."""
    chunks: list[ChunkMeta] = []
    for doc_id in index.doc_ids:
        meta = index.meta(doc_id)
        path = Path(meta.path)
        body = ""
        if path.is_file():
            try:
                body = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                body = ""
        full_text = f"{meta.title}\n{body}".strip()
        pieces = _split_chunks(full_text, chunk_words=chunk_words, overlap=overlap)
        if not pieces and meta.title:
            pieces = [meta.title]
        for cid, piece in enumerate(pieces):
            chunks.append(ChunkMeta(doc_id=doc_id, chunk_id=cid, text=piece))

    embeddings = encode_texts([c.text for c in chunks])
    return SemanticIndex(embeddings=embeddings, chunks=chunks)


def semantic_search(
    index: Index,
    sem_index: SemanticIndex,
    query: str,
    k: int = 10,
) -> list[SearchResult]:
    """Множить матрицю ембедингів на вектор запиту та зараховує документу найкращий фрагмент."""
    num_chunks = int(sem_index.embeddings.shape[0])
    if k <= 0 or num_chunks == 0 or not query.strip():
        return []

    emb_dim = int(sem_index.embeddings.shape[1])
    q_mat = encode_texts([query], dim=emb_dim)
    q_vec = q_mat.ravel()
    if float(np.linalg.norm(q_vec)) <= 1e-8:
        return []

    sims = (sem_index.embeddings @ q_vec).astype(np.float64)

    best_by_doc: dict[int, tuple[float, str]] = {}
    for idx_chunk, chunk in enumerate(sem_index.chunks):
        if chunk.doc_id not in index.doc_ids:
            continue
        score = float(sims[idx_chunk])
        prev = best_by_doc.get(chunk.doc_id)
        if prev is None or score > prev[0]:
            best_by_doc[chunk.doc_id] = (score, chunk.text)

    ranked = sorted(best_by_doc.items(), key=lambda kv: (-kv[1][0], kv[0]))[:k]
    return [
        SearchResult(
            score=score,
            doc_id=doc_id,
            title=index.meta(doc_id).title,
            snippet=best_chunk[:220],
        )
        for doc_id, (score, best_chunk) in ranked
    ]


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]],
    index: Index,
    k: int = 10,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Злиття кількох рейтингів через Reciprocal Rank Fusion: sum(1 / (rrf_k + rank))."""
    if k <= 0:
        return []

    rrf_scores: dict[int, float] = {}
    best_snippets: dict[int, str] = {}

    for ranking in rankings:
        for rank_idx, item in enumerate(ranking, start=1):
            rrf_scores[item.doc_id] = rrf_scores.get(item.doc_id, 0.0) + (1.0 / (rrf_k + rank_idx))
            if item.snippet and item.doc_id not in best_snippets:
                best_snippets[item.doc_id] = item.snippet

    merged = sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return [
        SearchResult(
            score=score,
            doc_id=doc_id,
            title=index.meta(doc_id).title,
            snippet=best_snippets.get(doc_id, ""),
        )
        for doc_id, score in merged
    ]


def hybrid_search(
    index: Index,
    sem_index: SemanticIndex | None,
    lexical_results: list[SearchResult],
    query: str,
    k: int = 10,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Зливає лексичний рейтинг (BM25) і семантичний рейтинг через RRF."""
    if sem_index is None:
        return lexical_results[:k]
    pool_k = max(k * 3, 20)
    sem_results = semantic_search(index, sem_index, query, k=pool_k)
    return reciprocal_rank_fusion([lexical_results, sem_results], index, k=k, rrf_k=rrf_k)
