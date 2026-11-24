"""FastAPI-застосунок findex: маршрути API, HTML-інтерфейс та обробка помилок."""

import logging
import math
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from ..model import Index, load_index
from ..query import parse
from ..scoring import make_scorer
from ..search import Matcher, SearchResult
from ..search import search as run_search
from ..semantic import (
    SearchMode,
    SemanticIndex,
    build_semantic_index,
    hybrid_search,
    semantic_search,
)
from ..snippets import snippet_for
from .config import Settings
from .schemas import (
    DocumentResponse,
    HealthResponse,
    SearchHitOut,
    SearchQueryParams,
    SearchResponse,
    StatsResponse,
)

log = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_MARK_START, _MARK_END = "\x02", "\x03"


def get_settings(request: Request) -> Settings:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings if settings is not None else Settings.from_env()


def get_index(request: Request) -> Index:
    index: Index | None = getattr(request.app.state, "index", None)
    if index is None or index.closed:
        raise HTTPException(status_code=503, detail="Індекс ще не завантажений.")
    return index


def get_matcher(request: Request, index: Annotated[Index, Depends(get_index)]) -> Matcher:
    matcher: Matcher | None = getattr(request.app.state, "matcher", None)
    if matcher is None:
        matcher = Matcher(index)
        request.app.state.matcher = matcher
    return matcher


def get_semantic_index(request: Request) -> SemanticIndex | None:
    return getattr(request.app.state, "semantic_index", None)


def _render_highlighted_html(raw_snippet: str) -> str:
    """Екранує HTML і перетворює парні маркери підсвітки у тег <mark>."""
    parts = raw_snippet.split(_MARK_START)
    first_part = next(iter(parts), "")
    out: list[str] = [escape(first_part)]
    for piece in parts[1:]:
        if _MARK_END in piece:
            hit, rest = piece.split(_MARK_END, 1)
            out.append(f"<mark>{escape(hit)}</mark>{escape(rest)}")
        else:
            out.append(escape(piece))
    return "".join(out)


def _safe_snippet(index: Index, doc_id: int, terms: list[str]) -> str:
    try:
        return snippet_for(index.meta(doc_id), terms, mark=(_MARK_START, _MARK_END))
    except OSError:
        return ""


def _read_document_text(path_str: str) -> str:
    path = Path(path_str)
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
    return ""


def _execute_search(
    index: Index,
    matcher: Matcher,
    params: SearchQueryParams,
    sem_index: SemanticIndex | None = None,
) -> SearchResponse:
    started = time.perf_counter()
    score_fn = make_scorer(params.scorer)

    if params.mode == "semantic":
        if sem_index is None:
            raise HTTPException(
                status_code=503, detail="Семантичний індекс (.npy) не завантажений."
            )
        all_results = semantic_search(index, sem_index, params.q, k=index.num_docs)
        terms = [w.lower() for w in params.q.split() if len(w) >= 2]
    elif params.mode == "hybrid":
        try:
            parsed_query = parse(params.q)
            lex_all = run_search(
                index, params.q, scorer=score_fn, k=index.num_docs, matcher=matcher
            )
            terms = list(dict.fromkeys(parsed_query.terms()))
        except ValueError:
            lex_all = []
            terms = [w.lower() for w in params.q.split() if len(w) >= 2]
        active_sem = sem_index if sem_index is not None else build_semantic_index(index)
        all_results = hybrid_search(index, active_sem, lex_all, params.q, k=index.num_docs)
    else:
        try:
            parsed_query = parse(params.q)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        all_matched = matcher(params.q)
        total_hits = len(all_matched)
        total_pages = max(1, math.ceil(total_hits / params.k)) if total_hits else 0

        if total_hits == 0 or params.page > total_pages:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return SearchResponse(
                query=params.q,
                scorer=params.scorer,
                mode=params.mode,
                page=params.page,
                per_page=params.k,
                total_hits=total_hits,
                total_pages=total_pages,
                elapsed_ms=round(elapsed_ms, 2),
                results=[],
            )

        needed = params.page * params.k
        top_slice = run_search(index, params.q, scorer=score_fn, k=needed, matcher=matcher)
        start_idx = (params.page - 1) * params.k
        page_items = top_slice[start_idx : start_idx + params.k]
        terms = list(dict.fromkeys(parsed_query.terms()))
        hits_out = _format_hits(index, page_items, start_idx, terms)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return SearchResponse(
            query=params.q,
            scorer=params.scorer,
            mode=params.mode,
            page=params.page,
            per_page=params.k,
            total_hits=total_hits,
            total_pages=total_pages,
            elapsed_ms=round(elapsed_ms, 2),
            results=hits_out,
        )

    total_hits = len(all_results)
    total_pages = max(1, math.ceil(total_hits / params.k)) if total_hits else 0
    if total_hits == 0 or params.page > total_pages:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return SearchResponse(
            query=params.q,
            scorer=params.scorer,
            mode=params.mode,
            page=params.page,
            per_page=params.k,
            total_hits=total_hits,
            total_pages=total_pages,
            elapsed_ms=round(elapsed_ms, 2),
            results=[],
        )

    start_idx = (params.page - 1) * params.k
    page_items = all_results[start_idx : start_idx + params.k]
    hits_out = _format_hits(index, page_items, start_idx, terms)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return SearchResponse(
        query=params.q,
        scorer=params.scorer,
        mode=params.mode,
        page=params.page,
        per_page=params.k,
        total_hits=total_hits,
        total_pages=total_pages,
        elapsed_ms=round(elapsed_ms, 2),
        results=hits_out,
    )


def _format_hits(
    index: Index,
    page_items: list[SearchResult],
    start_idx: int,
    terms: list[str],
) -> list[SearchHitOut]:
    hits_out: list[SearchHitOut] = []
    for offset, item in enumerate(page_items):
        meta = index.meta(item.doc_id)
        raw_snip = _safe_snippet(index, item.doc_id, terms)
        clean_snip = raw_snip.replace(_MARK_START, "").replace(_MARK_END, "")
        if not clean_snip and item.snippet:
            clean_snip = item.snippet
        hits_out.append(
            SearchHitOut(
                rank=start_idx + offset + 1,
                doc_id=item.doc_id,
                title=item.title or Path(meta.path).name,
                path=meta.path,
                score=round(item.score, 6),
                snippet=clean_snip,
            )
        )
    return hits_out


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings if settings is not None else Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        application.state.settings = cfg
        application.state.index = None
        application.state.matcher = None
        application.state.semantic_index = None

        if cfg.index_path is not None and cfg.index_path.is_file():
            log.info("Завантаження індексу із %s (%s)...", cfg.index_path, cfg.index_format)
            loaded = load_index(cfg.index_path, cfg.index_format)
            loaded.warm_numpy_cache()
            application.state.index = loaded
            application.state.matcher = Matcher(loaded)
            log.info("Індекс завантажено: %r", loaded)

            sem_candidate = (
                cfg.semantic_path
                if cfg.semantic_path is not None
                else cfg.index_path.with_suffix(".npy")
            )
            if (
                sem_candidate.with_suffix(".npy").is_file()
                and sem_candidate.with_suffix(".chunks.json").is_file()
            ):
                application.state.semantic_index = SemanticIndex.load(sem_candidate)
                log.info("Семантичний індекс завантажено із %s", sem_candidate)
            else:
                application.state.semantic_index = build_semantic_index(loaded)
                log.info("Семантичний індекс побудовано в пам'яті")
        else:
            log.warning("INDEX_PATH не задано або файл не знайдено: %s", cfg.index_path)

        try:
            yield
        finally:
            idx: Index | None = getattr(application.state, "index", None)
            if idx is not None and not idx.closed:
                idx.close()

    application = FastAPI(
        title="findex Web Search",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.settings = cfg

    @application.middleware("http")
    async def request_id_and_errors(  # type: ignore[reportUnusedFunction]
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = req_id
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        except Exception:
            log.exception("Неочікувана помилка під час запиту request_id=%s", req_id)
            return JSONResponse(
                status_code=500,
                content={"detail": "Внутрішня помилка сервера.", "request_id": req_id},
                headers={"X-Request-ID": req_id},
            )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> JSONResponse:  # type: ignore[reportUnusedFunction]
        index: Index | None = getattr(request.app.state, "index", None)
        sem_idx: SemanticIndex | None = getattr(request.app.state, "semantic_index", None)
        if index is None or index.closed:
            payload = HealthResponse(
                status="unavailable",
                index_loaded=False,
                semantic_loaded=False,
                num_docs=0,
            )
            return JSONResponse(status_code=503, content=payload.model_dump())
        payload = HealthResponse(
            status="ok",
            index_loaded=True,
            semantic_loaded=sem_idx is not None,
            num_docs=index.num_docs,
        )
        return JSONResponse(status_code=200, content=payload.model_dump())

    @application.get("/stats", response_model=StatsResponse)
    def stats(  # type: ignore[reportUnusedFunction]
        request: Request,
        index: Annotated[Index, Depends(get_index)],
    ) -> StatsResponse:
        sem_idx: SemanticIndex | None = getattr(request.app.state, "semantic_index", None)
        total_tokens = sum(index.doc_length(doc_id) for doc_id in index.doc_ids)
        top_terms = sorted(
            ((term, index.df(term)) for term in index),
            key=lambda item: (-item[1], item[0]),
        )[:10]
        return StatsResponse(
            num_docs=index.num_docs,
            num_terms=len(index),
            total_tokens=total_tokens,
            avg_doc_length=round(index.avg_doc_length, 2),
            has_positions=index.has_positions,
            semantic_loaded=sem_idx is not None,
            top_terms=top_terms,
        )

    @application.get("/search", response_model=SearchResponse)
    def api_search(  # type: ignore[reportUnusedFunction]
        request: Request,
        params: Annotated[SearchQueryParams, Query()],
        index: Annotated[Index, Depends(get_index)],
        matcher: Annotated[Matcher, Depends(get_matcher)],
    ) -> SearchResponse:
        sem_idx = get_semantic_index(request)
        return _execute_search(index, matcher, params, sem_index=sem_idx)

    @application.get("/docs/{doc_id}", response_model=DocumentResponse)
    def get_document(  # type: ignore[reportUnusedFunction]
        doc_id: int,
        index: Annotated[Index, Depends(get_index)],
    ) -> DocumentResponse:
        if doc_id not in index.doc_ids:
            raise HTTPException(status_code=404, detail=f"Документ doc_id={doc_id} не знайдено.")
        meta = index.meta(doc_id)
        return DocumentResponse(
            doc_id=doc_id,
            title=meta.title or Path(meta.path).name,
            path=meta.path,
            length=index.doc_length(doc_id),
            text=_read_document_text(meta.path),
        )

    @application.get("/", response_class=HTMLResponse)
    def home_page(  # type: ignore[reportUnusedFunction]
        request: Request,
        q: str = "",
        k: int = 10,
        scorer: str = "bm25",
        mode: str = "lexical",
        page: int = 1,
        doc_id: int | None = None,
    ) -> HTMLResponse:
        index: Index | None = getattr(request.app.state, "index", None)
        if index is None or index.closed:
            return templates.TemplateResponse(
                request,
                "index.html",
                {
                    "index_ready": False,
                    "q": q,
                    "k": k,
                    "scorer": scorer,
                    "mode": mode,
                    "page": page,
                    "search_data": None,
                    "highlighted_hits": [],
                    "error_msg": "Індекс ще не завантажений на сервері.",
                    "viewed_doc": None,
                },
                status_code=503,
            )

        matcher = get_matcher(request, index)
        sem_idx = get_semantic_index(request)
        viewed_doc: DocumentResponse | None = None
        if doc_id is not None and doc_id in index.doc_ids:
            meta = index.meta(doc_id)
            viewed_doc = DocumentResponse(
                doc_id=doc_id,
                title=meta.title or Path(meta.path).name,
                path=meta.path,
                length=index.doc_length(doc_id),
                text=_read_document_text(meta.path),
            )

        search_data: SearchResponse | None = None
        highlighted_hits: list[dict[str, object]] = []
        error_msg: str | None = None

        safe_mode: SearchMode = (
            mode if mode in ("lexical", "semantic", "hybrid") else "lexical"  # type: ignore[assignment]
        )

        cleaned_q = q.strip()
        if cleaned_q:
            safe_scorer: str = "tfidf" if scorer == "tfidf" else "bm25"
            try:
                params = SearchQueryParams(
                    q=cleaned_q,
                    k=max(1, min(100, k)),
                    scorer=safe_scorer,  # type: ignore[arg-type]
                    mode=safe_mode,
                    page=max(1, page),
                )
                search_data = _execute_search(index, matcher, params, sem_index=sem_idx)
                try:
                    parsed_terms = list(dict.fromkeys(parse(cleaned_q).terms()))
                except ValueError:
                    parsed_terms = [w.lower() for w in cleaned_q.split() if len(w) >= 2]
                for hit in search_data.results:
                    raw_s = _safe_snippet(index, hit.doc_id, parsed_terms)
                    snip_html = _render_highlighted_html(raw_s) if raw_s else escape(hit.snippet)
                    highlighted_hits.append(
                        {
                            "rank": hit.rank,
                            "doc_id": hit.doc_id,
                            "title": hit.title,
                            "score": hit.score,
                            "snippet_html": snip_html,
                        }
                    )
            except Exception as exc:
                error_msg = getattr(exc, "detail", str(exc))

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "index_ready": True,
                "num_docs": index.num_docs,
                "num_terms": len(index),
                "q": q,
                "k": k,
                "scorer": scorer,
                "mode": safe_mode,
                "page": page,
                "search_data": search_data,
                "highlighted_hits": highlighted_hits,
                "error_msg": error_msg,
                "viewed_doc": viewed_doc,
            },
        )

    return application


app = create_app()
