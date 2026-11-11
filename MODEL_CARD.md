# Model Card — Geo-Risk Q&A (RAG)

## What it is
A question-answering feature on top of the GDELT Geopolitical Risk Platform. It retrieves the most relevant spike-event and sector-reaction documents (built from this project's pipeline outputs) and asks Claude to answer **only** from those documents, citing each claim.

- **Retriever:** `sentence-transformers/all-MiniLM-L6-v2` embeddings + FAISS exact cosine search
- **Generator:** Claude via the Anthropic API (model set by `GEO_QA_MODEL`)
- **Knowledge source:** 31 detected geopolitical tension spikes (2016–2026) and the average returns of 6 groups of S&P 500 stocks at Day +1 / +5 / +30

## Intended use
- Exploring and explaining this project's historical results ("what happened around date X, and how did sector Y move afterwards?")
- Education, portfolio demonstration and research discussion

## Out of scope
- **Not investment advice.** The system describes historical averages only. It is instructed to decline buy/sell/hold questions and predictions, and it should not be used to make trading decisions.
- Questions about events, assets or periods outside the data (e.g. crypto, pre-2016, events that were not detected as spikes)
- Real-time news. The live feed in the dashboard is not part of the Q&A corpus.

## Data sources
The original cluster outputs are no longer accessible, so the knowledge base is **rebuilt** with `data_collection/rebuild_for_rag.py` for the same 31 spike dates. Spike-day Geo-Tension Index values are recomputed with the original formula; z-scores and thresholds are not available (details in the README's "Data provenance" section).

- [GDELT 2.0 GKG](https://www.gdeltproject.org/): news articles filtered to geopolitical themes, international coverage only
- S&P 500 daily closing prices via [yfinance](https://github.com/ranaroussi/yfinance)
- Event labels written by the project team (`dashboard/constants.py`); unlabeled spikes are named by their most frequent headline words

## Evaluation
Method: see "Evaluation method" in the README. The eval set is generated from the data (24 answerable and 5 unanswerable questions on the full rebuild).

> **Results: not run yet.** Paste the table from `rag/eval/results.md` after running `python -m rag.eval.run_eval`.

| Metric | Result |
|---|---|
| Retrieval hit rate @1 / @3 / @5 | — |
| Answer accuracy (answerable) | — |
| Grounded answers (auto-checkable) | — |
| Refusal accuracy (unanswerable) | — |
| False refusals (answerable) | — |

## Limitations and risks
- Rebuilt data: index values are recomputed for spike days only and may differ slightly from the original pipeline.
- Small corpus (31 events, ~50 stocks in 6 sectors), so coverage is narrow.
- Headlines are reconstructed from article URLs, not full text, and can be noisy.
- Spike labels are partly manual and partly keyword-based, and may oversimplify an event.
- Returns are averages over a few stocks per sector and don't control for other market news (correlation, not causation).
- Questions that need comparing all events can fail, because the LLM only sees the top k documents.
- The LLM can still misstate a number or cite the wrong document. The UI shows the retrieved documents so users can check, and the eval measures how often this happens.

## Safety measures
- The refusal gate skips the LLM call when retrieval similarity is below a threshold.
- The prompt restricts answers to the provided documents, requires citations and forbids investment advice.
- Code checks that every cited document was actually retrieved and flags any that weren't.
- The API key is read from an environment variable and never stored in the repo.
