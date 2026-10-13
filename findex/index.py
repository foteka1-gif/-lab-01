import argparse
import time
import os
import re
from collections import defaultdict, Counter
# Імпортуємо графічне вікно вибору файлу
from tkinter import filedialog, Tk 
from . import Posting, DocMeta
from .store import save_pickle, save_json

def clean_and_tokenize(text: str) -> list[str]:
    """Очищає текст від пунктуації та розбиває на слова."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

def build_index(corpus_pipeline):
    postings = defaultdict(list)
    doc_lengths = {}
    doc_meta = {}
    
    for doc_id, path, title, tokens in corpus_pipeline:
        doc_meta[doc_id] = DocMeta(path=path, title=title)
        doc_lengths[doc_id] = len(tokens)
        
        term_counts = Counter(tokens)
        for term, tf in term_counts.items():
            postings[term].append(Posting(doc_id=doc_id, tf=tf))
            
    return postings, doc_lengths, doc_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Побудова та збереження індексу.")
    parser.add_argument("--out", type=str, default="index.bin", help="Шлях до файлу для збереження індексу")
    parser.add_argument("--format", type=str, default="pickle", choices=["pickle", "json"], help="Формат збереження")
    args = parser.parse_args()
    
    # Створюємо приховане вікно комп'ютера для вибору файлу
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    print(" Будь ласка, виберіть ваш текстовий файл у вікні, що відкрилося...")
    # Відкриваємо провідник Windows для вибору файлу .txt
    file_path = filedialog.askopenfilename(
        title="Виберіть текстовий файл з Lab 01",
        filetypes=[("Текстові файли", "*.txt")]
    )
    
    if not file_path:
        print("❌ Файл не вибрано. Скрипт зупинено.")
        exit()
        
    print(f"✅ Вибрано файл: {file_path}")
    print("Будуємо індекс...")
    start_time = time.time()
    
    # Читаємо вибраний вами файл
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    filename = os.path.basename(file_path)
    title = os.path.splitext(filename)[0]
    tokens = clean_and_tokenize(content)
    
    # Передаємо в пайплайн наш єдиний реальний документ
    real_corpus = [(0, file_path, title, tokens)]
    index_data = build_index(real_corpus)
    
    build_elapsed = time.time() - start_time
    print(f"Індекс побудовано за {build_elapsed:.4f} сек.")
    print(f"Усього проіндексовано унікальних слів: {len(index_data[0])}")
    
    # Зберігаємо індекс поруч із вибраним файлом, щоб не загубити
    output_dir = os.path.dirname(file_path)
    final_output_path = os.path.join(output_dir, args.out)
    
    print(f"Зберігаємо індекс за шляхом: {final_output_path}...")
    if args.format == "pickle":
        save_pickle(index_data, final_output_path)
    else:
        save_json(index_data, final_output_path)
    print(" Успішно збережено!")
