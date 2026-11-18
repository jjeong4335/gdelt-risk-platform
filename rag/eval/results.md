# Geo-Risk Q&A eval — 2026-11-18

- Questions: 29 (24 answerable, 5 unanswerable)
- Retriever: **hybrid**; embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: gemini / `gemini-3.5-flash-lite`
- k given to LLM: 5; refusal similarity gate: 0.3

| Metric | Result |
|---|---|
| Retrieval hit rate @1 | 20/24 (83.3%) |
| Retrieval hit rate @3 | 23/24 (95.8%) |
| Retrieval hit rate @5 | 24/24 (100.0%) |
| Answer accuracy (answerable) | 24/24 (100.0%) |
| Grounded answers (auto-checkable) | 24/24 (100.0%) |
| Answers citing a non-retrieved doc | 0/24 (0.0%) |
| Refusal accuracy (unanswerable) | 5/5 (100.0%) |
| False refusals (answerable) | 0/24 (0.0%) |

Best cosine similarity in top 5 — answerable: min 0.346, mean 0.619; unanswerable: max 0.582, mean 0.496

## Failed or unchecked questions
