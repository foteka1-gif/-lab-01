import json
import pickle
 
from . import Posting, DocMeta
 
 
def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
 
 
def load_pickle(path):
    # Завантажувати можна тільки довірені Pickle-файли.
    with open(path, "rb") as f:
        return pickle.load(f)
 
 
def save_json(index_data, path):
    postings, doc_lengths, doc_meta = index_data
 
    def posting_to_dict(p):
        item = {"doc_id": p.doc_id, "tf": p.tf}
        if p.positions:  # ЛАБ 3: позиції пишемо, лише якщо вони є
            item["positions"] = list(p.positions)
        return item
 
    full_data = {
        "postings": {
            term: [posting_to_dict(p) for p in posting_list]
            for term, posting_list in postings.items()
        },
        "doc_lengths": {str(d): n for d, n in doc_lengths.items()},
        "doc_meta": {
            str(d): {"path": m.path, "title": m.title} for d, m in doc_meta.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
 
 
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
 
    postings = {
        term: [
            Posting(
                doc_id=p["doc_id"],
                tf=p["tf"],
                positions=tuple(p.get("positions", ())),
            )
            for p in raw_list
        ]
        for term, raw_list in full_data["postings"].items()
    }
    doc_lengths = {int(k): v for k, v in full_data["doc_lengths"].items()}
    doc_meta = {
        int(k): DocMeta(path=v["path"], title=v["title"])
        for k, v in full_data["doc_meta"].items()
    }
    return postings, doc_lengths, doc_meta
