import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from itertools import islice

try:
    from findex.corpus import iter_documents
    from findex.tokenize import tokenize
except ImportError:
    from corpus import iter_documents
    from tokenize import tokenize

def run_lazy_pipeline(data_path: Path, limit: int = None):
    docs_stream = iter_documents(data_path)
    if limit is not None:
        docs_stream = islice(docs_stream, limit)
    tokens_stream = (tok for doc in docs_stream for tok in tokenize(doc.text))
    return Counter(tokens_stream)

def run_eager_pipeline(data_path: Path, limit: int = None):
    docs_stream = list(iter_documents(data_path))
    if limit is not None:
        docs_stream = docs_stream[:limit]
    all_tokens = []
    for doc in docs_stream:
        all_tokens.extend(list(tokenize(doc.text)))
    return Counter(all_tokens)

def main():
    # Чітко вказуємо шлях до папки data у поточній робочій директорії
    data_path = Path("data")
    
    print(f"Шукаємо текстові файли у папці: {data_path.resolve()}")
    
    # Перевіримо, чи є взагалі файли
    txt_files = list(data_path.glob("*.txt"))
    print(f"Знайдено .txt файлів: {len(txt_files)}")
    for f in txt_files:
        print(f"  - {f.name}")

    mode = "lazy"
    limit = None
    
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]

    def measure(func, name):
        tracemalloc.start()
        start_time = time.perf_counter()
        
        counts = func(data_path, limit)
        
        elapsed = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_tokens = sum(counts.values())
        vocab_size = len(counts)
        
        print(f"\n=== Результати [{name.upper()}] ===")
        print(f"Загальна кількість токенів: {total_tokens}")
        print(f"Розмір словника (Vocabulary): {vocab_size}")
        print(f"Пікова пам'ять (Peak Memory): {peak / 1024 / 1024:.4f} MB")
        print(f"Витрачений час (Elapsed Time): {elapsed:.4f} s")
        print("Топ-10 слів:")
        for word, count in counts.most_common(10):
            print(f"  {word}: {count}")
            
    if mode in ["lazy", "both"]:
        measure(run_lazy_pipeline, "lazy (генератори)")
    if mode in ["eager", "both"]:
        measure(run_eager_pipeline, "eager (списки)")

if __name__ == "__main__":
    main()
