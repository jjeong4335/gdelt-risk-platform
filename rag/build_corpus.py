"""Step 1 — Build the text corpus from the EXISTING pipeline outputs.

This does NOT re-run Spark. It only reads the Parquet files the dashboard
already uses (in GEO_DATA_DIR, default ~/dashboard_data) and writes one JSON
line per "document" to rag/data/corpus.jsonl.

Chunking choice (why two kinds of documents):
  * one EVENT document per spike  -> answers "what happened on date X?",
    "how big was the spike?", "which sector fell most?"
  * one EVENT x SECTOR document   -> answers "how did airlines react to X?"
Each document is kept short (well under the embedding model's 256-token
limit) so nothing gets silently cut off when it is embedded.

Run (where the data lives, e.g. on the Dataproc edge node):
    python -m rag.build_corpus
    python -m rag.build_corpus --data-dir /path/to/dashboard_data
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import pandas as pd

from dashboard.constants import SECTOR_MAP, SPIKE_LABELS
from rag import config

# The pipeline outputs this script needs, relative to the data folder.
REQUIRED_OUTPUTS = {
    "spike_events": "spike_events.parquet",
    "geo_tension": "geo_tension_index.parquet",
    "spike_news": "spike_news",
    "reactions": "ticker_reaction_by_spike",
}

# The three horizons the Q&A talks about (calendar days after the spike).
HORIZONS = [1, 5, 30]
# Kept small so an event document stays under the model's 256-token limit.
HEADLINES_PER_EVENT = 3

# GDELT uses FIPS 10-4 country codes (e.g. RS = Russia, UP = Ukraine).
FIPS_NAMES = {
    "RS": "Russia", "UP": "Ukraine", "IS": "Israel", "IR": "Iran", "CH": "China",
    "KN": "North Korea", "KS": "South Korea", "SY": "Syria", "IZ": "Iraq",
    "AF": "Afghanistan", "PK": "Pakistan", "IN": "India", "TU": "Turkey",
    "UK": "United Kingdom", "FR": "France", "GM": "Germany", "JA": "Japan",
    "TW": "Taiwan", "HK": "Hong Kong", "SA": "Saudi Arabia", "YM": "Yemen",
    "LE": "Lebanon", "EG": "Egypt", "SU": "Sudan", "CE": "Sri Lanka",
    "BO": "Belarus", "PL": "Poland", "VE": "Venezuela", "MX": "Mexico",
    "CA": "Canada", "AS": "Australia", "NI": "Nigeria", "LY": "Libya",
    "QA": "Qatar", "AE": "United Arab Emirates", "JO": "Jordan", "GZ": "Gaza Strip",
    "WE": "West Bank", "BM": "Myanmar", "RP": "Philippines", "VM": "Vietnam",
}

_STOPWORDS = {"the", "and", "for", "are", "was", "has", "had", "with", "that", "this",
              "from", "have", "been", "will", "its", "not", "but", "they", "their",
              "said", "says", "over", "after", "more", "also", "than", "into", "about"}


def _to_date_str(series):
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


def check_outputs(data_dir: Path) -> None:
    """Stop early, with a clear message, if a pipeline output is missing."""
    missing = [name for name in REQUIRED_OUTPUTS.values() if not (data_dir / name).exists()]
    if missing:
        sys.exit(
            f"Missing pipeline output(s) in {data_dir}: {', '.join(missing)}\n"
            "Point --data-dir (or GEO_DATA_DIR) at the folder the dashboard reads."
        )


def load_outputs(data_dir: Path) -> dict:
    out = {}
    for key, name in REQUIRED_OUTPUTS.items():
        out[key] = pd.read_parquet(data_dir / name)
    for key, col in [("spike_events", "date"), ("geo_tension", "date"),
                     ("spike_news", "spike_date"), ("reactions", "spike_date")]:
        out[key][col] = _to_date_str(out[key][col])
    return out


def yearly_zscores(geo_tension: pd.DataFrame) -> pd.DataFrame:
    """Z-score = how many standard deviations above that year's average.

    Spike detection used: index > yearly_mean + 3 * yearly_std, so every
    spike has z > 3. pandas .std() uses the same sample formula as Spark.
    """
    df = geo_tension[["date", "geo_tension_index"]].copy()
    df["year"] = df["date"].str[:4]
    stats = df.groupby("year")["geo_tension_index"].agg(["mean", "std"])
    df = df.join(stats, on="year")
    df["z_score"] = (df["geo_tension_index"] - df["mean"]) / df["std"]
    return df.set_index("date")[["z_score"]]


def clean_headlines(news: pd.DataFrame) -> pd.DataFrame:
    """Same quality filter the dashboard uses for URL-derived titles."""
    t = news["title"].fillna("")
    mask = (
        (t.str.len() > 20)
        & ~t.str.contains(r"#|\.com|\.org|\.net", regex=True)
        & ~t.str.contains(r"[A-Fa-f0-9]{6,}", regex=True)
        & t.str.contains(" ")
        & (t.str.replace(r"[^a-zA-Z]", "", regex=True).str.len() > 15)
    )
    return news[mask]


def top_countries(news: pd.DataFrame, n: int = 3) -> list:
    """Most frequent non-US country codes in the GDELT theme tags."""
    counts = collections.Counter()
    for themes in news["themes"].dropna():
        for code in set(re.findall(r"#([A-Z]{2})#", themes)):
            if code != "US":
                counts[code] += 1
    return [FIPS_NAMES.get(code, code) for code, _ in counts.most_common(n)]


def event_label(date: str, news: pd.DataFrame) -> str:
    """Use the hand-written label if there is one, otherwise top title words
    (the same fallback the dashboard uses)."""
    if date in SPIKE_LABELS:
        return SPIKE_LABELS[date]
    words = re.findall(r"[a-z]{4,}", " ".join(news["title"].dropna()).lower())
    top = [w for w, _ in collections.Counter(w for w in words if w not in _STOPWORDS).most_common(3)]
    return "Unlabeled spike (frequent headline words: " + ", ".join(top) + ")" if top else "Unlabeled spike"


def sector_reactions(reactions: pd.DataFrame, date: str):
    """Average return per sector at Day +1/+5/+30.

    Returns (table, reaction_date). Like the dashboard:
      * if the spike day was not a trading day, use the nearest spike date
        that has price data;
      * if Day +N was a weekend/holiday, use the nearest trading day and
        record which day was actually used.
    """
    available = reactions["spike_date"].unique()
    if len(available) == 0:
        return {}, None
    if date in set(available):
        reaction_date = date
    else:
        d = pd.Timestamp(date)
        reaction_date = min(available, key=lambda x: abs((pd.Timestamp(x) - d).days))
    rows = reactions[reactions["spike_date"] == reaction_date].copy()

    ticker_to_sector = {t: s for s, tickers in SECTOR_MAP.items() for t in tickers}
    rows["sector"] = rows["ticker"].map(ticker_to_sector)
    rows = rows.dropna(subset=["sector"])
    days = sorted(d for d in rows["days_from_spike"].unique() if d > 0)
    if not days:
        return {}, reaction_date

    table = {}
    for horizon in HORIZONS:
        used = min(days, key=lambda x: abs(x - horizon))
        at_day = rows[rows["days_from_spike"] == used]
        by_sector = at_day.groupby("sector").agg(ret=("avg_return_pct", "mean"),
                                                 n=("ticker", "nunique"))
        for sector, r in by_sector.iterrows():
            table.setdefault(sector, {})[horizon] = {
                "return_pct": round(float(r["ret"]), 2),
                "day_used": int(used),
                "n_tickers": int(r["n"]),
            }
    return table, reaction_date


def fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


def horizon_text(cell: dict, horizon: int) -> str:
    s = f"Day +{horizon}: {fmt_pct(cell['return_pct'])}"
    if cell["day_used"] != horizon:
        s += f" (nearest trading day, +{cell['day_used']})"
    return s


def build_documents(data: dict) -> list:
    z = yearly_zscores(data["geo_tension"])
    news_all = data["spike_news"]
    docs = []

    for _, ev in data["spike_events"].sort_values("date").iterrows():
        date = ev["date"]
        news = news_all[news_all["spike_date"] == date]
        label = event_label(date, news)
        headlines = (clean_headlines(news)
                     .assign(dist=lambda d: d["days_from_spike"].abs())
                     .sort_values("dist")
                     .drop_duplicates("title")
                     .head(HEADLINES_PER_EVENT))
        urls = headlines["url"].tolist()
        source_url = urls[0] if urls else ""
        countries = top_countries(news)
        z_score = float(z.loc[date, "z_score"]) if date in z.index else float("nan")
        sectors, reaction_date = sector_reactions(data["reactions"], date)

        # ── EVENT document ──
        parts = [
            f"Event: {label}. Spike date: {date}.",
            f"Geo-Tension Index: {ev['geo_tension_index']:.2f} out of 10 "
            f"(z-score {z_score:.2f} versus the {date[:4]} daily average; "
            f"spike threshold {ev['threshold']:.2f}).",
            f"Dominant news category: {ev['dominant_category']}. "
            f"International articles that day: {int(ev['total_events']):,}.",
        ]
        if countries:
            parts.append("Most-mentioned countries in coverage: " + ", ".join(countries) + ".")
        if len(headlines):
            parts.append("Sample headlines (taken from article URLs): " + "; ".join(
                f"\"{r.title}\" ({r.domain})"
                for r in headlines.itertuples()) + ".")
        if sectors:
            day5 = sorted(((s, v[5]) for s, v in sectors.items() if 5 in v),
                          key=lambda x: x[1]["return_pct"])
            parts.append("Sector average returns at Day +5 after the spike: " + ", ".join(
                f"{s} {fmt_pct(c['return_pct'])}" for s, c in day5) + ".")
            parts.append(f"Weakest sector at Day +5: {day5[0][0]}. Strongest: {day5[-1][0]}.")
            if reaction_date != date:
                parts.append(f"(Price reactions measured from {reaction_date}, the nearest trading day.)")
        docs.append({
            "doc_id": f"event:{date}",
            "type": "event",
            "event_date": date,
            "label": label,
            "sector": None,
            "source_url": source_url,
            "urls": urls,
            "geo_tension_index": round(float(ev["geo_tension_index"]), 2),
            "z_score": round(z_score, 2),
            "countries": countries,
            "text": " ".join(parts),
        })

        # ── EVENT x SECTOR documents ──
        day5_rank = sorted((s for s in sectors if 5 in sectors[s]),
                           key=lambda s: sectors[s][5]["return_pct"])
        for sector, cells in sectors.items():
            tickers = ", ".join(SECTOR_MAP[sector])
            moves = "; ".join(horizon_text(cells[h], h) for h in HORIZONS if h in cells)
            text = (
                f"Sector reaction: {sector} stocks after the spike on {date} ({label}). "
                f"Average return of S&P 500 members ({tickers}) measured from the "
                f"spike-day close: {moves}."
            )
            if sector in day5_rank:
                text += (f" At Day +5 this sector ranked {day5_rank.index(sector) + 1} "
                         f"of {len(day5_rank)} (1 = fell the most).")
            docs.append({
                "doc_id": f"sector:{date}:{sector}",
                "type": "event_sector",
                "event_date": date,
                "label": label,
                "sector": sector,
                "source_url": source_url,
                "urls": urls[:1],
                "returns": {str(h): c for h, c in cells.items()},
                "text": text,
            })
    return docs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--out", type=Path, default=config.CORPUS_PATH)
    args = ap.parse_args()

    data_dir = args.data_dir.expanduser()
    check_outputs(data_dir)
    docs = build_documents(load_outputs(data_dir))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    n_events = sum(d["type"] == "event" for d in docs)
    print(f"Wrote {len(docs)} documents ({n_events} events, "
          f"{len(docs) - n_events} event-sector) to {args.out}")


if __name__ == "__main__":
    main()
