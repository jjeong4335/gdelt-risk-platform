"""Step 3 — Find the documents most relevant to a question.

Two retrievers, combined ("hybrid search"):

  * Dense (embeddings): good at MEANING — "fell the most" ~ "weakest sector".
    Weak at exact tokens: to an embedding model "2024-05-02" and "2024-08-01"
    look almost the same, so date questions often missed (found by the eval).
  * BM25 (keywords): classic search-engine scoring. A word scores high if it
    appears in the document and is rare across the corpus, so an exact date
    or sector name in the question pulls up exactly those documents.

Reciprocal Rank Fusion (RRF) merges the two rankings:
    rrf(doc) = 1 / (60 + dense_rank) + 1 / (60 + bm25_rank)
It uses only ranks, so the two very different score scales don't need to be
calibrated against each other. 60 is the constant from the original RRF paper.

Set GEO_QA_RETRIEVER=dense to use embeddings only (to compare in the eval).

Because the corpus is small we score ALL documents and filter afterwards, so
a date/sector filter never hides a good match.
"""
import json
import math
import re
from collections import Counter
from dataclasses import dataclass

from rag import config
from rag.index import embed

RRF_K = 60
# Dates stay one token ("2024-05-02"); otherwise words and numbers.
_TOKEN_RE = re.compile(r"\d{4}-\d{2}-\d{2}|[a-z]+|\d+(?:\.\d+)?")
_STOP = {"the", "a", "an", "of", "on", "in", "after", "what", "which", "how", "did", "was",
         "were", "is", "to", "and", "for", "by", "most", "days", "day", "spike", "stocks"}


def tokenize(text):
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75), written out so every step is visible."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        n = len(self.docs)
        df = Counter(term for d in self.docs for term in d)
        # rare terms (an exact date) get a high weight, common ones ("sector") low
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query):
        terms = tokenize(query)
        out = []
        for d, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in terms:
                tf = d.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    s += self.idf[t] * tf * (self.k1 + 1) / norm
            out.append(s)
        return out


@dataclass
class Hit:
    doc: dict
    score: float      # cosine similarity (used by the refusal gate and shown in the UI)
    rank_score: float = 0.0  # what the ranking used (RRF in hybrid mode)


class Retriever:
    def __init__(self, mode=None):
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
        self.mode = mode or config.RETRIEVER
        self.bm25 = BM25([d["text"] for d in self.docs])

    def _keep(self, doc, start_date, end_date, sectors):
        if start_date and doc["event_date"] < start_date:
            return False
        if end_date and doc["event_date"] > end_date:
            return False
        if sectors and doc["sector"] is not None and doc["sector"] not in sectors:
            return False
        return True

    def search(self, question, k=config.TOP_K, start_date=None, end_date=None, sectors=None):
        """start_date / end_date: 'YYYY-MM-DD' strings (inclusive), or None.
        sectors: list of sector names; event-level documents are always kept
        because they summarise every sector."""
        n = len(self.docs)
        scores, ids = self.index.search(embed([question]), n)
        cosine = {int(i): float(s) for s, i in zip(scores[0], ids[0])}

        if self.mode == "dense":
            ranked = [(i, cosine[i]) for i in ids[0]]
        else:
            dense_rank = {int(i): r for r, i in enumerate(ids[0], 1)}
            bm = self.bm25.scores(question)
            bm_order = sorted(range(n), key=lambda i: -bm[i])
            bm_rank = {i: r for r, i in enumerate(bm_order, 1)}
            rrf = {i: 1 / (RRF_K + dense_rank[i]) + 1 / (RRF_K + bm_rank[i]) for i in range(n)}
            ranked = sorted(rrf.items(), key=lambda x: -x[1])

        hits = []
        for i, rank_score in ranked:
            doc = self.docs[int(i)]
            if not self._keep(doc, start_date, end_date, sectors):
                continue
            hits.append(Hit(doc=doc, score=cosine[int(i)], rank_score=float(rank_score)))
            if len(hits) == k:
                break
        return hits
