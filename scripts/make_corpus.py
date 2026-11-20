"""Робить із кількох книг великий корпус для бенчмарку.

4 книги = 4 документи, а паралелізм ділить роботу ПО ДОКУМЕНТАХ: більше 4 воркерів
нічого не дасть. Тому ріжемо кожну книгу на шматки (~50 КБ) і, за потреби, копіюємо
весь набір кілька разів, щоб індексація тривала секунди, а не соту частку секунди.

Приклад:
    python scripts/make_corpus.py data data_big --chars 50000 --copies 20
"""

import argparse
from collections.abc import Iterator
from pathlib import Path


def split_text(text: str, size: int) -> Iterator[str]:
    """Ріже текст на шматки ~size символів, не розриваючи слова посередині."""
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        yield text[start:end]
        start = end


def main() -> None:
    parser = argparse.ArgumentParser(description="Великий корпус із книг для бенчмарку.")
    parser.add_argument("src", type=Path, help="папка з .txt книгами")
    parser.add_argument("dst", type=Path, help="куди писати шматки (напр. data_big)")
    parser.add_argument("--chars", type=int, default=50_000, help="розмір шматка в символах")
    parser.add_argument("--copies", type=int, default=1, help="скільки разів повторити набір")
    args = parser.parse_args()

    books = sorted(args.src.glob("*.txt"))
    if not books:
        raise SystemExit(f"У {args.src} немає .txt файлів.")

    args.dst.mkdir(parents=True, exist_ok=True)
    written = 0
    total_chars = 0
    for copy in range(args.copies):
        for book_number, book in enumerate(books):
            text = book.read_text(encoding="utf-8", errors="replace")
            for part, piece in enumerate(split_text(text, args.chars)):
                name = f"book{book_number}_c{copy:02d}_{part:04d}.txt"
                (args.dst / name).write_text(piece, encoding="utf-8")
                written += 1
                total_chars += len(piece)

    print(f"Записано {written} файлів, {total_chars / 1_000_000:.1f} млн символів у {args.dst}")


if __name__ == "__main__":
    main()
