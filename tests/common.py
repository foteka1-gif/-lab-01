from findex.index import build_index
from findex.tokenize import tokenize


def make_index(docs: dict, positions=True):
    """docs: {title: text} -> Index у пам'яті."""
    pipeline = (
        (i, f"{title}.txt", title, tokenize(text))
        for i, (title, text) in enumerate(docs.items())
    )
    return build_index(pipeline, positions=positions)
