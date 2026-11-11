"""Step 3 — Find the documents most relevant to a question.

1. Embed the question with the same model used for the documents.
2. Score every document by cosine similarity.
3. Apply optional filters (date range, sectors) using the document metadata.
4. Return the top k.

Because the corpus is small we score ALL documents and filter afterwards.
That guarantees a filter never hides a good match (a common bug when you
filter only the top few results of an approximate search).
"""
import json
from dataclasses import dataclass

from rag import config
from rag.index import embed


@dataclass
class Hit:
    doc: dict
    score: float  # cosine similarity, higher = more relevant


class Retriever:
    def __init__(self):
        import faiss

        if not config.INDEX_PATH.exists():
            raise FileNotFoundError(f"{config.INDEX_PATH} not found — run `python -m rag.index` first.")
        self.index = faiss.read_index(str(config.INDEX_PATH))
        with open(config.INDEX_META_PATH) as f:
            meta = json.load(f)
        if meta["model"] != config.EMBED_MODEL:
            raise RuntimeError(f"Index was built with {meta['model']}, but config uses "
                               f"{config.EMBED_MODEL}. Re-run `python -m rag.index`.")
        self.docs = meta["docs"]

    def search(self, question, k=config.TOP_K, start_date=None, end_date=None, sectors=None):
        """start_date / end_date: 'YYYY-MM-DD' strings (inclusive), or None.
        sectors: list of sector names; event-level documents are always kept
        because they summarise every sector."""
        qvec = embed([question])
        scores, ids = self.index.search(qvec, len(self.docs))

        hits = []
        for score, i in zip(scores[0], ids[0]):
            doc = self.docs[i]
            if start_date and doc["event_date"] < start_date:
                continue
            if end_date and doc["event_date"] > end_date:
                continue
            if sectors and doc["sector"] is not None and doc["sector"] not in sectors:
                continue
            hits.append(Hit(doc=doc, score=float(score)))
            if len(hits) == k:
                break
        return hits
