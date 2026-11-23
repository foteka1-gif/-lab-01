"""Командний рядок findex: crawl / index / search / stats.

Правила: дані -> stdout, діагностика (логи, помилки, прогрес) -> stderr.
Очікувана помилка = один зрозумілий рядок і ненульовий код виходу, без стека.
"""

import asyncio
import json
import logging
import pickle
import re
import sys
from collections.abc import Generator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NamedTuple

import typer
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from .corpus import list_document_paths, write_jsonl_record
from .crawler import CrawlStats, crawl
from .fetch import DEFAULT_USER_AGENT
from .index import build_index, make_pipeline
from .model import Index, IndexFormat, open_index, save_index
from .parallel import ExecutorName, build_index_parallel
from .query import parse
from .scoring import ScorerName, make_scorer
from .search import SearchResult
from .search import search as run_search
from .snippets import snippet_for

log = logging.getLogger(__name__)

console = Console()  # stdout: результати
err_console = Console(stderr=True)  # stderr: помилки і прогрес

# Керівні символи для підсвітки: не збігаються з текстом документів і не є пробілами.
_MARK_START, _MARK_END = "\x02", "\x03"

app = typer.Typer(
    help="findex — веб-краулер та пошук за інвертованим індексом.",
    no_args_is_help=True,
)


class FormatChoice(StrEnum):
    pickle = "pickle"
    json = "json"


class ScorerChoice(StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"


class ExecutorChoice(StrEnum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


def _index_format(choice: FormatChoice) -> IndexFormat:
    return "json" if choice is FormatChoice.json else "pickle"


def _scorer_name(choice: ScorerChoice) -> ScorerName:
    return "tfidf" if choice is ScorerChoice.tfidf else "bm25"


def _executor_name(choice: ExecutorChoice) -> ExecutorName:
    if choice is ExecutorChoice.threads:
        return "threads"
    if choice is ExecutorChoice.processes:
        return "processes"
    return "serial"


# --- логування: налаштовується один раз, тут ---------------------------------
_HANDLER_NAME = "findex-cli"


def setup_logging(verbosity: int) -> None:
    """0 -> WARNING, -v -> INFO, -vv -> DEBUG. Усе йде в stderr."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    package_logger = logging.getLogger("findex")
    for handler in list(package_logger.handlers):
        if handler.get_name() == _HANDLER_NAME:
            package_logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    )
    package_logger.addHandler(handler)
    package_logger.setLevel(level)


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="-v: INFO, -vv: DEBUG (у stderr)."),
    ] = 0,
) -> None:
    setup_logging(verbose)


# --- помилки -----------------------------------------------------------------
def fail(message: str) -> typer.Exit:
    """Один рядок у stderr і код виходу 1. Використання: raise fail("...")."""
    err_console.print(f"[red]Помилка:[/red] {escape(message)}", soft_wrap=True)
    return typer.Exit(code=1)


@contextmanager
def _expected_errors(index_file: Path) -> Generator[None]:
    """Перетворює очікувані винятки на один зрозумілий рядок."""
    try:
        yield
    except FileNotFoundError:
        raise fail(f"Файл індексу не знайдено: {index_file}") from None
    except (pickle.UnpicklingError, json.JSONDecodeError, EOFError):
        raise fail(f"{index_file} не схожий на файл індексу findex.") from None
    except ValueError as error:  # зламаний запит, фраза без позицій
        raise fail(str(error)) from None


# --- findex crawl ------------------------------------------------------------
def _crawl_status_table(stats: CrawlStats, max_pages: int) -> Table:
    table = Table(title="Краулер findex", show_header=True, header_style="bold cyan")
    table.add_column("Зібрано", justify="right")
    table.add_column("Стор./с", justify="right")
    table.add_column("У роботі", justify="right")
    table.add_column("Помилки", justify="right")
    table.add_column("Черга", justify="right")
    table.add_row(
        f"{stats.fetched}/{max_pages}",
        f"{stats.pages_per_second:.2f}",
        str(stats.in_flight),
        str(stats.errors),
        str(stats.queued),
    )
    return table


async def _run_crawl_to_files(
    start_url: str,
    *,
    out_path: Path,
    log_path: Path,
    max_pages: int,
    concurrency: int,
    per_host: int,
    delay: float,
    timeout: float,
    same_domain: bool,
    respect_robots: bool,
    user_agent: str,
) -> CrawlStats:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    last_stats = CrawlStats()
    with (
        out_path.open("w", encoding="utf-8") as jsonl_out,
        log_path.open("w", encoding="utf-8") as log_out,
        Live(
            _crawl_status_table(last_stats, max_pages),
            console=err_console,
            refresh_per_second=8,
        ) as live,
    ):
        async for page in crawl(
            start_url,
            max_pages=max_pages,
            concurrency=concurrency,
            per_host_concurrency=per_host,
            min_delay_per_host=delay,
            timeout_seconds=timeout,
            same_domain_only=same_domain,
            respect_robots=respect_robots,
            user_agent=user_agent,
        ):
            last_stats = page.stats
            if page.error is None:
                write_jsonl_record(
                    jsonl_out,
                    url=page.final_url,
                    title=page.title,
                    text=page.text,
                    status=page.status,
                )
                log_out.write(
                    f"OK\tstatus={page.status}\tattempts={page.attempts}\t"
                    f"bytes={page.num_bytes}\ttime={page.elapsed:.3f}s\t{page.final_url}\n"
                )
            else:
                log_out.write(
                    f"ERR\tstatus={page.status}\tattempts={page.attempts}\t"
                    f"error={page.error}\ttime={page.elapsed:.3f}s\t{page.url}\n"
                )
            log_out.flush()
            live.update(_crawl_status_table(last_stats, max_pages))

    return last_stats


# --- findex index ------------------------------------------------------------
@app.command("crawl")
def crawl_cmd(
    start_url: Annotated[str, typer.Argument(help="Початковий URL для обходу.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Куди писати JSONL.")] = Path(
        "pages.jsonl"
    ),
    log_file: Annotated[Path, typer.Option("--log-file", help="Файл журналу статусів URL.")] = Path(
        "crawl.log"
    ),
    max_pages: Annotated[int, typer.Option("--max-pages", min=1, help="Максимум сторінок.")] = 50,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", min=1, help="Максимум одночасних запитів.")
    ] = 5,
    per_host: Annotated[
        int, typer.Option("--per-host", min=1, help="Одночасних запитів на один хост.")
    ] = 2,
    delay: Annotated[
        float, typer.Option("--delay", min=0.0, help="Мінімальна пауза (с) між запитами на хост.")
    ] = 0.1,
    timeout: Annotated[
        float, typer.Option("--timeout", min=0.1, help="Таймаут одного запиту в секундах.")
    ] = 10.0,
    same_domain: Annotated[
        bool,
        typer.Option("--same-domain/--any-domain", help="Залишатися в межах стартового домену."),
    ] = True,
    respect_robots: Annotated[
        bool, typer.Option("--robots/--no-robots", help="Враховувати robots.txt.")
    ] = True,
    user_agent: Annotated[
        str, typer.Option("--user-agent", help="Заголовок User-Agent.")
    ] = DEFAULT_USER_AGENT,
    debug_asyncio: Annotated[
        bool, typer.Option("--debug-asyncio", help="Увімкнути debug-режим циклу подій asyncio.")
    ] = False,
) -> None:
    """Асинхронно обійти вебсторінки та зберегти їх у JSONL."""
    try:
        stats = asyncio.run(
            _run_crawl_to_files(
                start_url,
                out_path=out,
                log_path=log_file,
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                delay=delay,
                timeout=timeout,
                same_domain=same_domain,
                respect_robots=respect_robots,
                user_agent=user_agent,
            ),
            debug=debug_asyncio,
        )
    except ValueError as error:
        raise fail(str(error)) from None

    console.print(
        f"Зібрано сторінок: [bold]{stats.fetched}[/bold] "
        f"(помилок: {stats.errors}, пропущено: {stats.skipped}, "
        f"швидкість: {stats.pages_per_second:.2f} стор./с) -> [bold]{escape(str(out))}[/bold]"
    )


# --- findex index ------------------------------------------------------------
@app.command("index")
def index_cmd(
    corpus: Annotated[Path, typer.Argument(help="Папка з .txt документами або .jsonl файл.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Куди зберегти індекс.")] = Path(
        "index.bin"
    ),
    positions: Annotated[
        bool, typer.Option("--positions", help="Зберігати позиції токенів (для фраз).")
    ] = False,
    limit: Annotated[
        int | None, typer.Option("--limit", min=1, help="Проіндексувати лише перші N документів.")
    ] = None,
    fmt: Annotated[FormatChoice, typer.Option("--format", help="Формат файлу.")] = (
        FormatChoice.pickle
    ),
    workers: Annotated[
        int, typer.Option("--workers", "-w", min=1, help="Скільки воркерів (потоків/процесів).")
    ] = 1,
    executor: Annotated[
        ExecutorChoice, typer.Option("--executor", help="Чим будувати: serial, threads, processes.")
    ] = ExecutorChoice.serial,
) -> None:
    """Побудувати індекс із папки .txt документів або з .jsonl файлу краулера."""
    if corpus.is_file():
        log.info("JSONL-файл документів: %s", corpus.resolve())
        built_index = build_index(make_pipeline(corpus, limit=limit), positions=positions)
        if built_index.num_docs == 0:
            raise fail(f"У файлі {corpus} немає документів для індексації.")
    elif corpus.is_dir():
        paths = list_document_paths(corpus)
        if limit is not None:
            paths = paths[:limit]
        if not paths:
            raise fail(f"У папці {corpus} немає .txt файлів.")

        log.info("Папка документів: %s", corpus.resolve())
        with Progress(
            SpinnerColumn(),
            TextColumn("Індексація"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=err_console,
        ) as progress:
            task = progress.add_task("index", total=len(paths))
            report = build_index_parallel(
                paths,
                workers=workers,
                executor=_executor_name(executor),
                positions=positions,
                progress=lambda done: progress.advance(task, done),
            )
        log.info(
            "executor=%s workers=%d wall=%.3f с cpu=%.3f с merge=%.3f с",
            executor.value,
            workers,
            report.wall_seconds,
            report.cpu_seconds,
            report.merge_seconds,
        )
        built_index = report.built_index
    else:
        raise fail(f"Джерело документів не знайдено: {corpus}")

    out.parent.mkdir(parents=True, exist_ok=True)
    save_index(built_index, out, _index_format(fmt))
    log.info("Позиції збережено: %s", built_index.has_positions)
    console.print(
        f"Індекс збережено: [bold]{escape(str(out))}[/bold] — {escape(repr(built_index))}"
    )


# --- findex search -----------------------------------------------------------
class Hit(NamedTuple):
    rank: int
    result: SearchResult
    snippet: str


def _safe_snippet(index: Index, doc_id: int, terms: list[str]) -> str:
    try:
        return snippet_for(index.meta(doc_id), terms, mark=(_MARK_START, _MARK_END))
    except OSError as error:
        log.warning("Сніпет недоступний для doc_id=%d: %s", doc_id, error)
        return ""


def _highlight(snippet: str) -> Text:
    """Парні мітки -> жирний жовтий. Через Text, а не markup: у документах бувають [ ]."""
    text = Text()
    for i, part in enumerate(re.split(f"[{_MARK_START}{_MARK_END}]", snippet)):
        text.append(part, style="bold yellow" if i % 2 else "")
    return text


@app.command("search")
def search_cmd(
    index_file: Annotated[Path, typer.Argument(help="Файл індексу.")],
    query: Annotated[str, typer.Argument(help='Запит: python AND "async await" NOT java.')],
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="Скільки результатів.")] = 10,
    scorer: Annotated[ScorerChoice, typer.Option("--scorer", help="Функція ранжування.")] = (
        ScorerChoice.bm25
    ),
    as_json: Annotated[
        bool, typer.Option("--json", help="JSON-рядки у stdout (для | jq).")
    ] = False,
    snippets: Annotated[bool, typer.Option("--snippets/--no-snippets")] = True,
    fmt: Annotated[FormatChoice, typer.Option("--format", help="Формат файлу.")] = (
        FormatChoice.pickle
    ),
) -> None:
    """Ранжований пошук за запитом."""
    with _expected_errors(index_file), open_index(index_file, _index_format(fmt)) as index:
        score_fn = make_scorer(_scorer_name(scorer))
        results = run_search(index, query, scorer=score_fn, k=k)
        terms = list(dict.fromkeys(parse(query).terms()))
        hits = [
            Hit(rank, res, _safe_snippet(index, res.doc_id, terms) if snippets else "")
            for rank, res in enumerate(results, start=1)
        ]

    log.info("Запит %r -> %d результатів", query, len(hits))

    if as_json:
        for hit in hits:
            payload = {
                "rank": hit.rank,
                "doc_id": hit.result.doc_id,
                "title": hit.result.title,
                "score": round(hit.result.score, 6),
                "snippet": re.sub(f"[{_MARK_START}{_MARK_END}]", "**", hit.snippet),
            }
            sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return

    if not hits:
        console.print("Нічого не знайдено.")
        return

    table = Table(title=f"Результати для {query!r} ({scorer.value})")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Оцінка", justify="right", style="green", no_wrap=True)
    table.add_column("Документ", style="bold")
    if snippets:
        table.add_column("Фрагмент")

    for hit in hits:
        row: list[str | Text] = [
            str(hit.rank),
            f"{hit.result.score:.4f}",
            escape(hit.result.title),
        ]
        if snippets:
            row.append(_highlight(hit.snippet))
        table.add_row(*row)

    console.print(table)


# --- findex stats ------------------------------------------------------------
@app.command("stats")
def stats_cmd(
    index_file: Annotated[Path, typer.Argument(help="Файл індексу.")],
    top: Annotated[int, typer.Option("--top", min=1, help="Скільки частих термів показати.")] = 10,
    fmt: Annotated[FormatChoice, typer.Option("--format", help="Формат файлу.")] = (
        FormatChoice.pickle
    ),
) -> None:
    """Статистика індексу."""
    with _expected_errors(index_file), open_index(index_file, _index_format(fmt)) as index:
        total_tokens = sum(index.doc_length(doc_id) for doc_id in index.doc_ids)
        summary = Table(title=f"Статистика: {index_file}")
        summary.add_column("Показник", style="bold")
        summary.add_column("Значення", justify="right")
        summary.add_row("Документів", str(index.num_docs))
        summary.add_row("Унікальних термів", str(len(index)))
        summary.add_row("Усього токенів", str(total_tokens))
        summary.add_row("Середня довжина документа", f"{index.avg_doc_length:.1f}")
        summary.add_row("Позиції збережено", "так" if index.has_positions else "ні")
        summary.add_row("Розмір файлу", f"{index_file.stat().st_size:,} байт")

        top_terms = sorted(
            ((term, index.df(term)) for term in index),
            key=lambda item: (-item[1], item[0]),
        )[:top]

        top_str = ", ".join(f"{term} ({df})" for term, df in top_terms)
        if top_str:
            summary.add_row("Часті терми", escape(top_str))

    console.print(summary)


if __name__ == "__main__":
    app()
