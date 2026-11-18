# Model Card — Geo-Risk Q&A (RAG)

## What it is
A question-answering feature on top of the GDELT Geopolitical Risk Platform. It retrieves the most relevant spike-event and sector-reaction documents (built from this project's pipeline outputs) and asks an LLM to answer **only** from those documents, citing each claim.

- **Retriever:** `sentence-transformers/all-MiniLM-L6-v2` embeddings + FAISS exact cosine search
- **Generator:** Gemini via Google AI Studio by default (`gemini-3.8-flash`), or Claude via the Anthropic API (`GEO_QA_PROVIDER`, `GEO_QA_MODEL`). The model used is printed in every eval report.
- **Knowledge source:** geopolitical tension spikes detected automatically in GDELT data (2016–2026) and the average returns of 6 groups of S&P 500 stocks at Day +1 / +5 / +30

## Intended use
- Exploring and explaining this project's historical results ("what happened around date X, and how did sector Y move afterwards?")
- Education, portfolio demonstration and research discussion

## Out of scope
- **Not investment advice.** The system describes historical averages only. It is instructed to decline buy/sell/hold questions and predictions, and it should not be used to make trading decisions.
- Questions about events, assets or periods outside the data (e.g. crypto, pre-2016, events that were not detected as spikes)
- Real-time news. The live feed in the dashboard is not part of the Q&A corpus.

## Data sources
Rebuilt from GDELT's public BigQuery tables after the original cluster outputs became unavailable. Spike dates, event names and categories are generated from the data (`build_history.py` → `build_events.py`); no event is labelled by hand.

- [GDELT 2.0 GKG](https://www.gdeltproject.org/): articles whose Themes match a geopolitical category and that mention a non-US country
- S&P 500 daily closing prices (today's constituents) via [yfinance](https://github.com/ranaroussi/yfinance)
- Event names: the most distinctive countries + category for each event, scored by lift against the other events (no LLM)

## Evaluation
Method: see "Evaluation" in the README. 29 questions generated from the data (24 answerable, 5 unanswerable). Run 2026-11-18, 27 events, LLM `gemini-3.5-flash-lite`, k = 5.

| Metric | Result |
|---|---|
| Retrieval hit rate @1 / @3 / @5 (hybrid) | 83.3% / 95.8% / 100.0% (20, 23, 24 of 24) |
| Answer accuracy (answerable) | 100.0% (24/24) |
| Grounded answers (numbers/dates found in retrieved docs) | 100.0% (24/24) |
| Refusal accuracy (unanswerable) | 100.0% (5/5) |
| False refusals (answerable) | 0.0% (0/24) |

The first run on the same 27 events scored 75.0% hit@5 and 75.0% answer accuracy; retrieval was changed after looking at those failures (see the README), so these numbers are optimistic. The set is small, and "grounded" means every number and date comes from the retrieved documents, not that the answer is correct.

## Limitations and risks
- Small corpus (a few dozen events, ~50 stocks in 6 sectors), so coverage is narrow.
- Headlines are reconstructed from article URLs, not full text, and can be noisy.
- Auto-generated names (countries + category) can oversimplify an event, and several events can share a name; the date tells them apart.
- Returns are averages over a few stocks per sector and don't control for other market news (correlation, not causation).
- Questions that compare all events rely on precomputed comparison documents; comparisons they don't cover can fail, because the LLM only sees the top k documents.
- The 0–10 index is capped, so several events tie at 10.00; the z-score is the better measure of size.
- The LLM can still misstate a number or cite the wrong document. The UI shows the retrieved documents so users can check, and the eval measures how often this happens.

## Safety measures
- The refusal gate skips the LLM call when retrieval similarity is below a threshold.
- The prompt restricts answers to the provided documents, requires citations and forbids investment advice.
- Code checks that every cited document was actually retrieved and flags any that weren't.
- The API key is read from an environment variable and never stored in the repo.
- On the Gemini free tier, prompts and responses may be used by Google to improve its products. Only public data (GDELT news metadata and stock prices) is sent.
