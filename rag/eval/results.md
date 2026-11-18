# Geo-Risk Q&A eval — 2026-11-17

- Questions: 28 (23 answerable, 5 unanswerable)
- Retriever: **hybrid**; embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: gemini / `gemini-3.5-flash-lite`
- k given to LLM: 5; refusal similarity gate: 0.3

| Metric | Result |
|---|---|
| Retrieval hit rate @1 | 13/23 (56.5%) |
| Retrieval hit rate @3 | 20/23 (87.0%) |
| Retrieval hit rate @5 | 20/23 (87.0%) |
| Answer accuracy (answerable) | 20/23 (87.0%) |
| Grounded answers (auto-checkable) | 21/21 (100.0%) |
| Answers citing a non-retrieved doc | 0/21 (0.0%) |
| Refusal accuracy (unanswerable) | 5/5 (100.0%) |
| False refusals (answerable) | 2/23 (8.7%) |

Best cosine similarity in top 5 — answerable: min 0.435, mean 0.637; unanswerable: max 0.675, mean 0.513

## Failed or unchecked questions
- **q13** What was the Geo-Tension Index on 2018-05-15? — expected doc not in top 5; expected '10.00'
  - answer: I don't have evidence for that in the data.
- **q16** What was the Geo-Tension Index on 2025-06-09? — expected doc not in top 5; expected '10.00'
  - answer: I don't have evidence for that in the data.
- **q23** After which spike did Airlines stocks fall the most by Day +5? — expected doc not in top 5; expected '2025-06-09'
  - answer: Airlines stocks fell the most by Day +5 after the geopolitical tension spike on 2025-06-10 (Iran War), with an average return of -4.97% [sector:2025-06-10:Airlines].
