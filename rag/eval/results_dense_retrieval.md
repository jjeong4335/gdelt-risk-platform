# Geo-Risk Q&A eval — 2026-11-18

- Questions: 29 (24 answerable, 5 unanswerable)
- Retriever: **dense**; embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: gemini / `gemini-3.5-flash-lite` (not used)
- k given to LLM: 5; refusal similarity gate: 0.3

| Metric | Result |
|---|---|
| Retrieval hit rate @1 | 18/24 (75.0%) |
| Retrieval hit rate @3 | 24/24 (100.0%) |
| Retrieval hit rate @5 | 24/24 (100.0%) |
| Generation metrics | not run (--retrieval-only) |

Best cosine similarity in top 5 — answerable: min 0.332, mean 0.620; unanswerable: max 0.602, mean 0.511

## Failed or unchecked questions
