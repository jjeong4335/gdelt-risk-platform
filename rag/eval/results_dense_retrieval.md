# Geo-Risk Q&A eval — 2026-11-17

- Questions: 28 (23 answerable, 5 unanswerable)
- Retriever: **dense**; embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: gemini / `gemini-3.5-flash-lite` (not used)
- k given to LLM: 5; refusal similarity gate: 0.3

| Metric | Result |
|---|---|
| Retrieval hit rate @1 | 9/23 (39.1%) |
| Retrieval hit rate @3 | 12/23 (52.2%) |
| Retrieval hit rate @5 | 13/23 (56.5%) |
| Generation metrics | not run (--retrieval-only) |

Best cosine similarity in top 5 — answerable: min 0.444, mean 0.639; unanswerable: max 0.686, mean 0.524

## Weak spots
  -> hit@5 is low (56.5%). Suggested fix: Add the event label and date to the start of every document, or add a keyword (BM25) retriever and combine its scores with the embeddings.

## Failed or unchecked questions
- **q02** Which sector fell the most 5 days after the spike on 2019-10-03? — expected doc not in top 5
- **q05** Which sector fell the most 5 days after the spike on 2024-05-02? — expected doc not in top 5
- **q06** Which sector fell the most 5 days after the spike on 2025-06-11? — expected doc not in top 5
- **q11** How did Semiconductors stocks move 30 days after the spike on 2024-08-01? — expected doc not in top 5
- **q12** How did Tech stocks move 30 days after the spike on 2025-06-12? — expected doc not in top 5
- **q13** What was the Geo-Tension Index on 2018-05-15? — expected doc not in top 5
- **q14** What was the Geo-Tension Index on 2022-02-18? — expected doc not in top 5
- **q15** What was the Geo-Tension Index on 2024-04-26? — expected doc not in top 5
- **q16** What was the Geo-Tension Index on 2025-06-09? — expected doc not in top 5
- **q23** After which spike did Airlines stocks fall the most by Day +5? — expected doc not in top 5
