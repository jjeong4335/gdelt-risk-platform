"""
GDELT DOC API-based real-time RAG system with FAISS.
For a given spike date + category, fetches live news articles
and ranks them by semantic similarity.
"""

import time
import requests
import numpy as np
import faiss
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

# ── Constants ────────────────────────────────────────────────────────────────
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MODEL_NAME    = "all-MiniLM-L6-v2"
WINDOW_DAYS   = 3
MAX_RECORDS   = 50

CATEGORY_QUERIES = {
    "POLITICAL_INSTABILITY": "political unrest government crisis protest",
    "CONFLICT":              "military conflict war attack violence",
    "SANCTIONS":             "economic sanctions trade restrictions",
    "PROTEST":               "protest demonstration civil unrest",
    "DIPLOMACY":             "diplomatic relations international agreement",
    "ECONOMIC_CRISIS":       "economic crisis financial market collapse",
}

# Load model once at module level (reused across calls)
_model = None

def get_model():
    """Lazy-load sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def fetch_gdelt_news(query: str, date: str,
                     window_days: int = WINDOW_DAYS,
                     max_records: int = MAX_RECORDS,
                     retry: int = 3) -> list[dict]:
    """
    Fetch English news articles from GDELT DOC API around a spike date.
    Retries on 429 with exponential backoff.
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

    for attempt in range(retry):
        try:
            r = requests.get(GDELT_API_URL, params=params, timeout=30)
            if r.status_code == 429:
                wait = 5 * (2 ** attempt)
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()

            articles = r.json().get("articles", [])
            results  = []
            for a in articles:
                if a.get("language", "").lower() != "english":
                    continue
                results.append({
                    "title":      a.get("title", ""),
                    "url":        a.get("url", ""),
                    "date":       a.get("seendate", ""),
                    "domain":     a.get("domain", ""),
                    "country":    a.get("sourcecountry", ""),
                    "spike_date": date,
                })
            return results

        except Exception as e:
            print(f"API error attempt {attempt+1}: {e}")
            time.sleep(5)

    return []


def query_realtime(spike_date: str, category: str,
                   top_k: int = 5) -> list[dict]:
    """
    Real-time RAG query:
      1. Fetch live articles from GDELT API for the spike date
      2. Embed article titles with sentence-transformers
      3. Rank by cosine similarity to category query
      4. Return top-k results

    Args:
        spike_date: 'YYYY-MM-DD' format
        category:   Dominant event category from spike_events
        top_k:      Number of top articles to return

    Returns:
        List of article dicts with similarity scores
    """
    query_text = CATEGORY_QUERIES.get(category, "geopolitical crisis conflict")
    articles   = fetch_gdelt_news(query_text, spike_date)

    if not articles:
        return []

    model      = get_model()
    texts      = [a["title"] for a in articles]
    embeddings = model.encode(texts, normalize_embeddings=True)
    q_vec      = model.encode([query_text], normalize_embeddings=True)

    # Build temporary FAISS index for this query
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    scores, indices = index.search(q_vec.astype(np.float32), min(top_k, len(articles)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        article = articles[idx].copy()
        article["similarity"] = float(score)
        results.append(article)

    return results


if __name__ == "__main__":
    print("=== Testing Real-time RAG Query ===")

    # Test: Russia-Ukraine spike
    results = query_realtime(
        spike_date="2022-02-24",
        category="CONFLICT",
        top_k=5
    )

    print(f"\nTop {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['similarity']:.4f}")
        print(f"    Title:  {r['title']}")
        print(f"    Source: {r['domain']} ({r['country']})")
        print(f"    URL:    {r['url']}")
