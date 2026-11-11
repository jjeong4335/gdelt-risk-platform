"""Step 2 — Turn each document into a vector and store the vectors.

An "embedding" is a list of 384 numbers that captures what a sentence means.
Sentences with similar meaning get vectors that point in similar directions,
so "Which sector fell most after the Ukraine invasion?" lands close to the
document about the 2022-02-25 Russia-Ukraine spike.

We normalise every vector to length 1, so the inner product FAISS computes
equals cosine similarity (1.0 = same meaning, ~0 = unrelated).

Index type: IndexFlatIP = exact search over every vector. With a few hundred
documents this takes microseconds, so approximate indexes would add
complexity for no gain.

Run:  python -m rag.index
"""
import json

import numpy as np

from rag import config

_model = None


def get_model():
    """Load the embedding model once (downloads ~80 MB the first time)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed(texts):
    vecs = get_model().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def load_corpus(path=config.CORPUS_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m rag.build_corpus` first.")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index():
    import faiss

    docs = load_corpus()
    model = get_model()

    # Warn if any document is longer than the model can read.
    max_len = model.max_seq_length
    too_long = [d["doc_id"] for d in docs
                if len(model.tokenizer.tokenize(d["text"])) > max_len]
    if too_long:
        print(f"WARNING: {len(too_long)} documents exceed {max_len} tokens and will be "
              f"truncated when embedded, e.g. {too_long[:3]}")

    vecs = embed(d["text"] for d in docs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    with open(config.INDEX_META_PATH, "w") as f:
        json.dump({"model": config.EMBED_MODEL, "docs": docs}, f)
    print(f"Indexed {len(docs)} documents ({vecs.shape[1]}-dim) -> {config.INDEX_PATH}")


if __name__ == "__main__":
    build_index()
