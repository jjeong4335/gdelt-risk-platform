# Model Card — Geo-Risk Q&A (RAG)

## What it is
A question-answering feature on top of the GDELT Geopolitical Risk Platform. It retrieves the most relevant spike-event and sector-reaction documents (built from this project's pipeline outputs) and asks an LLM to answer **only** from those documents, citing each claim.

- **Retriever:** `sentence-transformers/all-MiniLM-L6-v2` embeddings + FAISS exact cosine search
- **Generator:** Gemini via Google AI Studio by default (`gemini-3.8-flash`), or Claude via the Anthropic API (`GEO_QA_PROVIDER`, `GEO_QA_MODEL`). The model used is printed in every eval report.
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
Method: see "Evaluation method" in the README. The eval set is generated from the data (23 answerable and 5 unanswerable questions on the full rebuild).

Run 2026-11-17 on the full rebuild; LLM `gemini-3.5-flash-lite`, k = 5.

| Metric | Result |
|---|---|
| Retrieval hit rate @1 / @3 / @5 (hybrid) | 56.5% / 87.0% / 87.0% (13, 20, 20 of 23) |
| Retrieval hit rate @5, dense-only baseline | 56.5% (13/23) |
| Answer accuracy (answerable) | 87.0% (20/23) |
| Grounded answers (numbers/dates found in retrieved docs) | 100.0% (21/21) |
| Refusal accuracy (unanswerable) | 100.0% (5/5) |
| False refusals (answerable) | 8.7% (2/23) |

The set is small (23 answerable questions), so treat these as indicative. "Grounded" means every number and date in the answer appears in the retrieved documents, not that the answer is correct. See the README for what failed and why.

## Limitations and risks
- Rebuilt data: index values are recomputed for spike days only and may differ slightly from the original pipeline.
- Small corpus (31 events, ~50 stocks in 6 sectors), so coverage is narrow.
- Headlines are reconstructed from article URLs, not full text, and can be noisy.
- Spike labels are partly manual and partly keyword-based, and may oversimplify an event.
- Returns are averages over a few stocks per sector and don't control for other market news (correlation, not causation).
- Questions that need comparing all events can fail, because the LLM only sees the top k documents.
- Recomputed index values are capped at 10.00, so several spikes tie at the top.
- The LLM can still misstate a number or cite the wrong document. The UI shows the retrieved documents so users can check, and the eval measures how often this happens.

## Safety measures
- The refusal gate skips the LLM call when retrieval similarity is below a threshold.
- The prompt restricts answers to the provided documents, requires citations and forbids investment advice.
- Code checks that every cited document was actually retrieved and flags any that weren't.
- The API key is read from an environment variable and never stored in the repo.
- On the Gemini free tier, prompts and responses may be used by Google to improve its products. Only public data (GDELT news metadata and stock prices) is sent.
