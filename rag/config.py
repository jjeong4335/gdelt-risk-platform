"""All paths and tunable settings for the Q&A feature live here.

Every value can be overridden with an environment variable, so nothing
(especially the API key) is hard-coded.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = REPO_ROOT / "rag"

# Where the existing pipeline outputs live (same folder the dashboard reads).
DATA_DIR = Path(os.environ.get("GEO_DATA_DIR", "~/dashboard_data")).expanduser()

# Output of build_corpus.py. Small (a few hundred documents), so it is
# committed to git: the index and eval can then run on a laptop without
# access to the cluster data.
CORPUS_PATH = RAG_DIR / "data" / "corpus.jsonl"

# Output of index.py. Rebuilt in seconds, so it is git-ignored.
ARTIFACT_DIR = RAG_DIR / "artifacts"
INDEX_PATH = ARTIFACT_DIR / "index.faiss"
INDEX_META_PATH = ARTIFACT_DIR / "index_meta.json"

# Embedding model: small (80 MB), runs on CPU, good general-purpose
# English sentence similarity. Max input length is 256 tokens, which is why
# each document is kept short (see build_corpus.py).
EMBED_MODEL = os.environ.get("GEO_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LLM used to write the answer. The API key is read by the anthropic SDK
# from the ANTHROPIC_API_KEY environment variable.
LLM_MODEL = os.environ.get("GEO_QA_MODEL", "claude-sonnet-5-5")

# "hybrid" = embeddings + BM25 keyword search merged with RRF (default);
# "dense" = embeddings only. See rag/retrieve.py.
RETRIEVER = os.environ.get("GEO_QA_RETRIEVER", "hybrid")

# How many documents to give the LLM.
TOP_K = int(os.environ.get("GEO_QA_TOP_K", "5"))

# If even the best-matching document has cosine similarity below this,
# we refuse without calling the LLM ("no evidence in the data").
# Tune it with the numbers printed by rag/eval/run_eval.py.
MIN_SIMILARITY = float(os.environ.get("GEO_QA_MIN_SIM", "0.30"))

# The exact sentence the system uses when the data has no answer.
# The eval script looks for this sentence to score refusals.
REFUSAL_TEXT = "I don't have evidence for that in the data."
