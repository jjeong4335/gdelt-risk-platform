# Legacy: original course pipeline (Spring 2026)

The original NYU Big Data team project ran on the NYU Dataproc cluster:
`gdelt_download.sh` streamed and filtered the GDELT GKG archive into HDFS, and the
PySpark jobs in `pyspark_pipeline/` built the daily index, spike windows and news.
Cluster access ended with the course, so this code is kept for reference only and
is not maintained. `rebuild_for_rag.py` was an intermediate laptop rebuild of the
original 31 hand-picked spike dates; it no longer runs (those labels were removed).

The current pipeline (BigQuery → automatic spike detection and naming → RAG) is in
`data_collection/` and `rag/`; see the main README. Problems found in this code and
fixed there:

- Keywords were matched against GKG column 7 (V2Counts) instead of Themes, so every
  day's dominant category came out the same.
- Spikes were detected on the 0–10 index, which is capped at 10; in 2017–2020 the
  yearly mean + 3σ threshold was above 10, so no spike could be detected.
- Spike dates and event names were picked by hand.
- A spike on a weekend had no day-0 close, and the dashboard borrowed another spike's
  numbers.
