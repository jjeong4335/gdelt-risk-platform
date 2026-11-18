"""STEP 4 of the data-driven rebuild: name each event and attach news + prices.

Inputs (in ~/dashboard_data_rebuilt/):
    events.csv              detected events      (from build_history.py)
    event_news.csv          news per event       (from event_news.sql in BigQuery)
    sp500_prices.parquet    daily closes         (from download_prices.py)

Every event name is generated from the data, not typed by hand:
    "<top countries>: <category>"   e.g. "Ukraine & Russia: conflict"
using the countries mentioned most in that event's articles and the most
common article category. The top news themes are kept as evidence.

Outputs (the files the dashboard and RAG code read):
    spike_events.parquet, spike_news/part-0.parquet,
    ticker_reaction_by_spike/part-0.parquet, ticker_summary.parquet

Run (from the repo root):
    python -m data_collection.build_events
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from data_collection.rebuild_for_rag import build_reactions, url_to_title
from rag.build_corpus import FIPS_NAMES

CATEGORY_WORDS = {
    "CONFLICT": "conflict",
    "ECONOMIC_PRESSURE": "sanctions & trade",
    "POLITICAL_INSTABILITY": "unrest",
    "DIPLOMATIC_TENSION": "diplomatic tension",
    "DEESCALATION": "ceasefire",
    "INTELLIGENCE_CYBER": "espionage & cyber",
    "SPECIAL": "nuclear & blockade",
}
# Generic GDELT theme families that say little about what happened.
GENERIC_THEME_PREFIXES = ("TAX_FNCACT", "TAX_ETHNICITY", "TAX_WORLDLANGUAGES", "TAX_POLITICAL_PARTY",
                          "CRISISLEX", "WB_", "EPU_", "UNGP_", "SOC_", "USPEC_", "GENERAL_", "MEDIA_",
                          "LEADER", "GOV", "MANMADE", "ECON_", "TAX_ECON", "EDUCATION", "TAX_DISEASE")


def parse_counts(s):
    """'RS:120|UP:95' -> [('RS', 120), ('UP', 95)]"""
    out = []
    for part in str(s or "").split("|"):
        if ":" in part:
            k, v = part.rsplit(":", 1)
            out.append((k, int(v)))
    return out


def name_event(countries, category):
    """Top country, plus the second one if it is mentioned at least 40% as often."""
    names = [FIPS_NAMES.get(c, c) for c, _ in countries]
    if not names:
        main = "International"
    elif len(countries) > 1 and countries[1][1] >= 0.4 * countries[0][1]:
        main = f"{names[0]} & {names[1]}"
    else:
        main = names[0]
    return f"{main}: {CATEGORY_WORDS.get(category, 'geopolitical tension')}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("~/dashboard_data_rebuilt"))
    args = ap.parse_args()
    d = args.data_dir.expanduser()

    for f in ("events.csv", "event_news.csv"):
        if not (d / f).exists():
            raise SystemExit(f"{d / f} not found — see the steps in the README (Data provenance).")
    events = pd.read_csv(d / "events.csv", parse_dates=["date", "start_date", "end_date"])
    news = pd.read_csv(d / "event_news.csv", parse_dates=["event_date", "article_date"])

    rows, news_parts = [], []
    for ev in events.itertuples():
        n = news[news["event_date"] == ev.date]
        countries = parse_counts(n["top_countries"].iloc[0]) if len(n) else []
        themes = [t for t, _ in parse_counts(n["top_themes"].iloc[0])] if len(n) else []
        themes = [t for t in themes if not t.startswith(GENERIC_THEME_PREFIXES)][:6]
        category = Counter(n["category"]).most_common(1)[0][0] if len(n) else ev.dominant_category
        rows.append({
            "date": ev.date.date(), "start_date": ev.start_date.date(), "end_date": ev.end_date.date(),
            "spike_days": ev.spike_days, "geo_tension_index": ev.geo_tension_index, "z_score": ev.z_score,
            "total_events": ev.total_events, "is_spike": True,
            "dominant_category": category,
            "label": name_event(countries[:2], category),
            "top_countries": ", ".join(FIPS_NAMES.get(c, c) for c, _ in countries[:5]),
            "top_themes": ", ".join(themes),
            "articles_in_window": int(n["articles_in_window"].iloc[0]) if len(n) else 0,
        })
        if len(n):
            news_parts.append(pd.DataFrame({
                "spike_date": ev.date.date(), "dominant_category": category,
                "article_date": n["article_date"].dt.date, "domain": n["domain"], "url": n["url"],
                "themes": n["category"], "days_from_spike": n["days_from_spike"].astype(int),
                "title": n["url"].map(url_to_title),
            }))

    spike_events = pd.DataFrame(rows)
    spike_events.to_parquet(d / "spike_events.parquet")
    spike_news = pd.concat(news_parts, ignore_index=True).dropna(subset=["title"])
    (d / "spike_news").mkdir(exist_ok=True)
    spike_news.to_parquet(d / "spike_news" / "part-0.parquet")

    print(f"{len(spike_events)} events named from the data:")
    for r in spike_events.itertuples():
        print(f"  {r.date}  {r.label:45s} z={r.z_score:.1f}  themes: {r.top_themes[:60]}")
    dupes = spike_events["label"].value_counts()
    if (dupes > 1).any():
        print("  (same name used by more than one event — the date tells them apart: "
              + ", ".join(dupes[dupes > 1].index) + ")")

    prices_path = d / "sp500_prices.parquet"
    if not prices_path.exists():
        print(f"\n{prices_path} not found — run python -m data_collection.download_prices, then re-run this.")
        return
    reactions = build_reactions(pd.read_parquet(prices_path), dates=[str(x) for x in spike_events["date"]])
    reactions["dominant_category"] = reactions["spike_date"].map(dict(zip(spike_events["date"],
                                                                          spike_events["dominant_category"])))
    (d / "ticker_reaction_by_spike").mkdir(exist_ok=True)
    reactions.to_parquet(d / "ticker_reaction_by_spike" / "part-0.parquet")
    g = reactions.groupby("ticker")
    pd.DataFrame({
        "avg_return_5d": reactions[reactions["days_from_spike"] == 5].groupby("ticker")["avg_return_pct"].mean(),
        "avg_return_30d": reactions[reactions["days_from_spike"] == 30].groupby("ticker")["avg_return_pct"].mean(),
        "worst_drawdown": g["avg_return_pct"].min(),
        "num_events": g["spike_date"].nunique(),
    }).reset_index().to_parquet(d / "ticker_summary.parquet")
    print(f"\nStock reactions: {reactions['ticker'].nunique()} tickers x "
          f"{reactions['spike_date'].nunique()} events = {len(reactions):,} rows")
    print(f"\nNext: python -m rag.build_corpus --data-dir {d}")


if __name__ == "__main__":
    main()
