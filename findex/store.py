import pickle
import json
import sys
from . import Posting, DocMeta

# Додатково реєструємо класи у головному просторі імен для pickle
import __main__
__main__.Posting = Posting
__main__.DocMeta = DocMeta

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_json(index_data, path):
    postings, doc_lengths, doc_meta = index_data
    
    serializable_postings = {}
    for term, posting_list in postings.items():
        serializable_postings[term] = [{"doc_id": p.doc_id, "tf": p.tf} for p in posting_list]
        
    serializable_meta = {}
    for doc_id, meta in doc_meta.items():
        serializable_meta[str(doc_id)] = {"path": meta.path, "title": meta.title}
        
    serializable_lengths = {str(k): v for k, v in doc_lengths.items()}
    
    full_data = {
        "postings": serializable_postings,
        "doc_lengths": serializable_lengths,
        "doc_meta": serializable_meta
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
        
    postings = {}
    for term, raw_list in full_data["postings"].items():
        postings[term] = [Posting(doc_id=p["doc_id"], tf=p["tf"]) for p in raw_list]
        
    doc_lengths = {int(k): v for k, v in full_data["doc_lengths"].items()}
    
    doc_meta = {}
    for k, v in full_data["doc_meta"].items():
        doc_meta[int(k)] = DocMeta(path=v["path"], title=v["title"])
        
    return postings, doc_lengths, doc_meta
