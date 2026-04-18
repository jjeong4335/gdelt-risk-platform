# Geopolitical Event-Driven Portfolio Risk Analysis Platform

> "VIX tells you how scared the market is. This system tells you why — and how exposed your portfolio is to that cause."

## Overview

A large-scale data pipeline that processes 10 years of GDELT global news records alongside S&P 500 price data to:

- Compute a daily **Geo-Tension Index** from global news sentiment
- Detect geopolitical tension spike events and analyze market reactions in **±30-day windows**
- Build a dataset of **sector and ticker-level reaction patterns** per event type
- Accept a user portfolio as input and return a **historical risk summary**
- Explain *why* a risk score is high via a **RAG-based news evidence layer**
- Refresh automatically every 15 minutes with a **live Streamlit dashboard**

No machine learning. No prediction. Pure large-scale data engineering and retrieval — fully explainable outputs.

---

## Team

| Name | NetID |
|------|-------|
| David Hong | sh8348 |
| Jonghyun Jeong | jj4335 |
| Tinos Vafias | cv2134 |

---

## Architecture

```
GDELT Archive (2016–2025)          S&P 500 Price Data (yfinance)
        |                                       |
        v                                       v
  Download + Filter                     Chunk collection
  (simultaneous, xargs -P 4)            (Parquet format)
        |                                       |
        +---------------+---------------+
                        |
                        v
              HDFS: /user/jj4335_nyu_edu/gdelt_project/
                        |
                        v
                 PySpark ETL Pipeline
                 - VADER sentiment scoring
                 - Daily Geo-Tension Index
                 - Tension spike detection
                 - ±30-day event window aggregation
                        |
              +---------+---------+
              |                   |
              v                   v
     Reaction Pattern         FAISS Vector Index
       Dataset (Parquet)       (RAG layer)
              |                   |
              +---------+---------+
                        |
                        v
              Portfolio Risk Engine
              (weighted historical summary)
                        |
                        v
              Streamlit Dashboard
              (APScheduler: 15-min refresh)
```

---

## Repository Structure

```
gdelt-risk-platform/
├── data_collection/
│   └── gdelt_download.sh       # Download + filter GDELT data simultaneously
├── pyspark_pipeline/
│   └── geo_tension_index.py    # PySpark ETL: Geo-Tension Index + event windows
├── rag/
│   └── faiss_index.py          # Embed news articles + FAISS vector search
├── dashboard/
│   └── app.py                  # Streamlit dashboard + APScheduler
└── README.md
```

---

## Infrastructure

| Component | Tool |
|-----------|------|
| Cluster | NYU Dataproc (Google Cloud Dataproc) |
| Resource Manager | YARN |
| Distributed Processing | Apache Spark (PySpark) |
| Storage | HDFS (`/user/jj4335_nyu_edu/gdelt_project/`) |
| NLP / Sentiment | VADER |
| Scheduling | APScheduler |
| Vector Search (RAG) | FAISS |
| Dashboard | Streamlit + Plotly |
| Version Control | Git / GitHub |

---

## Data Sources

| Source | Description | Cost |
|--------|-------------|------|
| [GDELT Project](https://gdeltproject.org) | Global news events, sentiment (2016–present) | Free |
| [yfinance](https://github.com/ranaroussi/yfinance) | S&P 500 daily prices, 10 years | Free |
| [OFAC Sanctions](https://ofac.treasury.gov) | US sanctions event list | Free |

---

## Data Collection Strategy

We tested four approaches for 1 month of GDELT data (January 2016):

| Method | Time | Size |
|--------|------|------|
| Download only | 15 min | 26 GB |
| Download then filter (sequential) | 44 min | 58 MB |
| Filter only (pre-downloaded) | 29 min | 58 MB |
| **Download + filter simultaneously** ✅ | **5 min** | **58 MB** |

**Final strategy:** Stream directly from GDELT → filter geopolitical keywords on-the-fly → write to HDFS. No intermediate 26 GB storage required.

Geopolitical keywords: `MILITARY`, `SANCTION`, `TERROR`, `CONFLICT`, `WAR`, `PROTEST`, `WEAPON`, `NUCLEAR`

10-year dataset: ~7 GB filtered, stored at `/user/jj4335_nyu_edu/gdelt_project/gdelt/` with 755 permissions (owner write, team read-only).

---

## Why PySpark?

| Task | Why Spark? |
|------|-----------|
| Filter keywords from hundreds of millions of records | Single machine cannot load the full GDELT archive |
| 500 tickers × all events × ±30-day windows | Distributed join across partitioned datasets |
| Sector-level aggregations across 500 tickers | Parallel groupBy + window operations |
| Daily tension index from massive text corpus | Distributed aggregation at scale |

---

## RAG: Explainability Layer

The Geo-Tension Index answers *"how dangerous?"* — but users need to know *"why?"*

For each tension spike, the RAG layer retrieves the top-K most relevant GDELT news articles from a FAISS vector index and surfaces them alongside the quantitative risk score.

**Example output:**
```
Geo-Tension Index: 0.73 (High)
Portfolio Risk: Energy sector -12% expected

"During Iran-Israel tensions in 2024, 23 Reuters/Bloomberg articles
 reported an avg -11.8% drop in energy stocks, with refiners most affected."
```

---
