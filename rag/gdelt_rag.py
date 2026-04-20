mkdir -p ~/gdelt-risk-platform/rag
cat > ~/gdelt-risk-platform/rag/gdelt_rag.py << 'EOF'
"""
GDELT DOC API-based RAG system with FAISS for spike event news retrieval.
Fetches real-time English news articles, embeds them, and enables semantic search.
"""

import os
import json
import pickle
import requests
import numpy as np
import pandas as pd
import faiss
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

# ── Constants ────────────────────────────────────────────────────────────────
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
INDEX_DIR = os.path.expanduser("~/gdelt-risk-platform/rag/index")
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, lightweight, good quality
WINDOW_DAYS = 3                   # ±3 days around spike date
MAX_RECORDS = 50                  # Articles per API call

# Category → search query mapping
CATEGORY_QUERIES = {
    "POLITICAL_INSTABILITY": "political unrest government crisis protest",
    "CONFLICT":              "military conflict war attack violence",
    "SANCTIONS":             "economic sanctions trade restrictions",
    "PROTEST":               "protest demonstration civil unrest",
    "DIPLOMACY":             "diplomatic relations international agreement",
    "ECONOMIC_CRISIS":       "economic crisis financial market collapse",
}

os.makedirs(INDEX_DIR, exist_ok=True)


# ── GDELT Fetcher ─────────────────────────────────────────────────────────────
def fetch_gdelt_news(query: str, date: str, window_days: int = WINDOW_DAYS,
                     max_records: int = MAX_RECORDS) -> list[dict]:
    """
    Fetch English news articles from GDELT DOC API around a spike date.

    Args:
        query:       Search query string
        date:        Spike date in 'YYYY-MM-DD' format
        window_days: Days before/after spike to search
        max_records: Max articles to retrieve

    Returns:
        List of article dicts with title, url, date, domain
    """
    spike_dt = datetime.strptime(date, "%Y-%m-%d")
    start_dt = spike_dt - timedelta(days=window_days)
    end_dt   = spike_dt + timedelta(days=window_days)

    params = {
        "query":         f"{query} sourcelang:english",
        "mode":          "artlist",
        "maxrecords":    max_records,
        "startdatetime": start_dt.strftime("%Y%m%d000000"),
        "enddatetime":   end_dt.strftime("%Y%m%d235959"),
        "format":        "json",
    }

    try:
        r = requests.get(GDELT_API_URL, params=params, timeout=30)
        r.raise_for_status()
        articles = r.json().get("articles", [])

        # Filter English only and extract relevant fields
        results = []
        for a in articles:
            if a.get("language", "").lower() != "english":
                continue
            results.append({
                "title":   a.get("title", ""),
                "url":     a.get("url", ""),
                "date":    a.get("seendate", ""),
                "domain":  a.get("domain", ""),
                "country": a.get("sourcecountry", ""),
                "query":   query,
                "spike_date": date,
            })

        print(f"  Fetched {len(results)} English articles for '{query}' @ {date}")
        return results

    except Exception as e:
        print(f"  API error for '{query}' @ {date}: {e}")
        return []


# ── FAISS Index Builder ───────────────────────────────────────────────────────
def build_index(spike_events_path: str, force_rebuild: bool = False):
    """
    Build FAISS index from spike events using GDELT DOC API.

    Args:
        spike_events_path: Path to spike_events.parquet
        force_rebuild:     Rebuild even if index already exists
    """
    index_path    = os.path.join(INDEX_DIR, "faiss.index")
    metadata_path = os.path.join(INDEX_DIR, "metadata.pkl")

    # Load existing index if available
    if not force_rebuild and os.path.exists(index_path) and os.path.exists(metadata_path):
        print("Index already exists. Loading from disk...")
        index    = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        print(f"Loaded {len(metadata)} articles from existing index.")
        return index, metadata

    print("Building FAISS index from GDELT DOC API...")
    model  = SentenceTransformer(MODEL_NAME)
    spikes = pd.read_parquet(spike_events_path)

    all_articles = []

    for _, row in spikes.iterrows():
        date     = str(row["date"])[:10]
        category = row.get("dominant_category", "POLITICAL_INSTABILITY")
        query    = CATEGORY_QUERIES.get(category, "geopolitical crisis conflict")

        print(f"\nSpike: {date} | Category: {category}")
        articles = fetch_gdelt_news(query, date)
        all_articles.extend(articles)

    if not all_articles:
        raise ValueError("No articles fetched. Check API connectivity.")

    # Build text for embedding: title is the primary signal
    texts = [a["title"] for a in all_articles]

    print(f"\nEmbedding {len(texts)} articles with {MODEL_NAME}...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True)

    # Build FAISS index (inner product = cosine similarity after normalization)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    # Save index and metadata
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(all_articles, f)

    print(f"\nIndex built: {len(all_articles)} articles, dim={dim}")
    print(f"Saved to {INDEX_DIR}")
    return index, all_articles


# ── RAG Query Engine ──────────────────────────────────────────────────────────
class GDELTRagEngine:
    """
    RAG engine: given a spike date + category, retrieve semantically
    relevant news articles using FAISS similarity search.
    """

    def __init__(self, spike_events_path: str, force_rebuild: bool = False):
        self.model    = SentenceTransformer(MODEL_NAME)
        self.index, self.metadata = build_index(spike_events_path, force_rebuild)

    def query(self, spike_date: str, category: str, top_k: int = 5,
              realtime: bool = True) -> list[dict]:
        """
        Retrieve top-k relevant articles for a spike event.

        Args:
            spike_date: Date string 'YYYY-MM-DD'
            category:   Dominant event category
            top_k:      Number of results to return
            realtime:   If True, also fetch fresh articles from API

        Returns:
            List of top-k article dicts with similarity scores
        """
        query_text = CATEGORY_QUERIES.get(category, "geopolitical crisis conflict")

        # Optionally fetch fresh articles in real-time
        if realtime:
            fresh = fetch_gdelt_news(query_text, spike_date)
            if fresh:
                texts      = [a["title"] for a in fresh]
                embeddings = self.model.encode(texts, normalize_embeddings=True)
                self.index.add(embeddings.astype(np.float32))
                self.metadata.extend(fresh)

        # Embed query and search
        q_vec = self.model.encode([query_text], normalize_embeddings=True)
        scores, indices = self.index.search(q_vec.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            article = self.metadata[idx].copy()
            article["similarity"] = float(score)
            results.append(article)

        return results

    def query_realtime(self, spike_date: str, category: str,
                       top_k: int = 5) -> list[dict]:
        """
        Pure real-time query: fetch fresh articles from API and rank by
        embedding similarity without using the pre-built index.

        Args:
            spike_date: Date string 'YYYY-MM-DD'
            category:   Dominant event category
            top_k:      Number of results to return

        Returns:
            List of top-k ranked article dicts
        """
        query_text = CATEGORY_QUERIES.get(category, "geopolitical crisis conflict")
        articles   = fetch_gdelt_news(query_text, spike_date, max_records=50)

        if not articles:
            return []

        texts      = [a["title"] for a in articles]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        q_vec      = self.model.encode([query_text], normalize_embeddings=True)

        # Cosine similarity (embeddings are normalized)
        scores = (embeddings @ q_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            article = articles[idx].copy()
            article["similarity"] = float(scores[idx])
            results.append(article)

        return results


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SPIKE_EVENTS_PATH = os.path.expanduser(
        "~/gdelt-risk-platform/dashboard_data/spike_events.parquet"
        if os.path.exists(os.path.expanduser(
            "~/gdelt-risk-platform/dashboard_data/spike_events.parquet"))
        else "~/dashboard_data/spike_events.parquet"
    )

    print("=== Testing Real-time RAG Query ===")
    engine = GDELTRagEngine(SPIKE_EVENTS_PATH, force_rebuild=False)

    # Test with Russia-Ukraine spike
    results = engine.query_realtime(
        spike_date="2022-02-24",
        category="CONFLICT",
        top_k=5
    )

    print("\nTop results:")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['similarity']:.4f}")
        print(f"    Title:  {r['title']}")
        print(f"    Source: {r['domain']} ({r['country']})")
        print(f"    URL:    {r['url']}")
EOF
echo "Done!"
