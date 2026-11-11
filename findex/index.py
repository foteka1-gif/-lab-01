import argparse
import os
import time
import tracemalloc
from collections import defaultdict, Counter
from pathlib import Path
 
from . import Posting, DocMeta
from .corpus import iter_documents
from .decorators import timed, setup_logging
from .model import Index, save_index, load_index
from .tokenize import tokenize
 
 
@timed  # ЛАБ 3: сліди часу побудови в лозі
def build_index(corpus_pipeline, positions=False):
    """Тепер повертає Index (а не кортеж із трьох dict-ів)."""
    postings = defaultdict(list)
    doc_lengths = {}
    doc_meta = {}
 
    for doc_id, path, title, tokens in corpus_pipeline:
        doc_meta[doc_id] = DocMeta(path=str(path), title=title)
 
        if positions:
            term_counts = Counter()
            term_positions = defaultdict(list)
            for pos, token in enumerate(tokens):
                term_counts[token] += 1
                term_positions[token].append(pos)
        else:
            term_counts = Counter(tokens)
            term_positions = None
 
        doc_lengths[doc_id] = sum(term_counts.values())
 
        for term, tf in term_counts.items():
            postings[term].append(
                Posting(
                    doc_id=doc_id,
                    tf=tf,
                    positions=tuple(term_positions[term]) if positions else (),
                )
            )
 
    return Index(dict(postings), doc_lengths, doc_meta)
 
 
def make_pipeline(root):
    for doc_id, doc in enumerate(iter_documents(root)):
        filename = Path(doc.path).name
        title = Path(filename).stem
        tokens = tokenize(doc.text)
        yield doc_id, doc.path, title, tokens
 
 
def main():
    parser = argparse.ArgumentParser(description="Побудова інвертованого індексу.")
    parser.add_argument("data", type=str, nargs="?", default="data")
    parser.add_argument("--out", type=str, default="index.bin")
    parser.add_argument("--format", choices=["pickle", "json"], default="pickle")
    parser.add_argument("--memory-report", action="store_true")
    parser.add_argument("--positions", action="store_true",
                        help="зберігати позиції токенів (потрібно для фраз)")
    args = parser.parse_args()
 
    setup_logging()
 
    root = Path(args.data)
    if not root.exists() or not root.is_dir():
        parser.error("Папка з документами не знайдена.")
 
    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
 
    print("Папка документів:", root.resolve())
    print("Починаємо побудову індексу...")
 
    tracemalloc.start()
    start_time = time.perf_counter()
    index = build_index(make_pipeline(root), positions=args.positions)
    build_time = time.perf_counter() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
 
    print("\nІНДЕКС ПОБУДОВАНО")
    print("-------------------------")
    print(index)
    print("Документів:", index.num_docs)
    print("Унікальних слів:", len(index))
    print("Усього постінгів:", sum(len(v) for v in index.values()))
    print("Позиції збережено:", index.has_positions)
    print(f"Середня довжина документа: {index.avg_doc_length:.1f} токенів")
    print(f"Час побудови: {build_time:.6f} сек.")
    print(f"Поточна пам'ять: {current_memory / 1024 / 1024:.4f} MB")
    print(f"Пікова пам'ять: {peak_memory / 1024 / 1024:.4f} MB")
 
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    print("\nЗБЕРЕЖЕННЯ ІНДЕКСУ")
    save_start = time.perf_counter()
    save_index(index, output_path, args.format)
    save_time = time.perf_counter() - save_start
    print("Файл:", output_path)
    print(f"Розмір файлу: {os.path.getsize(output_path) / 1024 / 1024:.4f} MB")
    print(f"Час збереження: {save_time:.6f} сек.")
 
    print("\nЗАВАНТАЖЕННЯ ІНДЕКСУ")
    load_start = time.perf_counter()
    loaded = load_index(output_path, args.format)
    load_time = time.perf_counter() - load_start
    print(f"Час завантаження: {load_time:.6f} сек.")
    print("Індекс успішно завантажено:", loaded)
 
    # Index — це Mapping, тому benchmark.py працює з ним без жодних змін логіки
    if args.memory_report:
        from .benchmark import measure_memory_report
        measure_memory_report(index)
 
    from .benchmark import benchmark_search
    benchmark_search(index)
 
    print("\nРОБОТУ ЗАВЕРШЕНО УСПІШНО.")
 
 
if __name__ == "__main__":
    main()