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
from pathlib import Path

import numpy as np
import pandas as pd

from data_collection.common import build_reactions, url_to_title
from dashboard.constants import FIPS_NAMES

CATEGORY_WORDS = {
    "CONFLICT": "conflict",
    "ECONOMIC_PRESSURE": "sanctions & trade",
    "POLITICAL_INSTABILITY": "unrest",
    "DIPLOMATIC_TENSION": "diplomatic tension",
    "DEESCALATION": "ceasefire",
    "INTELLIGENCE_CYBER": "espionage & cyber",
    "SPECIAL": "nuclear & blockade",
    "TERRORISM": "terror attack",
}
# Generic GDELT theme families that say little about what happened.
GENERIC_THEME_PREFIXES = ("TAX_FNCACT", "TAX_ETHNICITY", "TAX_WORLDLANGUAGES", "TAX_POLITICAL_PARTY",
                          "CRISISLEX", "WB_", "EPU_", "UNGP_", "SOC_", "USPEC_", "GENERAL_", "MEDIA_",
                          "LEADER", "GOV", "MANMADE", "ECON_", "TAX_ECON", "EDUCATION", "TAX_DISEASE",
                          "TAX_WORLDMAMMALS", "TAX_WORLDBIRDS", "AFFECT", "MEDICAL")


def parse_counts(s):
    """'RS:120|UP:95' -> [('RS', 120), ('UP', 95)]"""
    out = []
    for part in str(s or "").split("|"):
        if ":" in part:
            k, v = part.rsplit(":", 1)
            out.append((k, int(v)))
    return out


def shares_and_lift(count_strings, totals):
    """Rows = events, columns = items (countries / categories / themes).
    share = fraction of the event's articles that mention the item;
    lift  = share / the item's average share over all events.
    Lift is what makes an event different: the UK, Russia, CONFLICT and
    ARMEDCONFLICT appear around almost every event, so raw counts would put
    them on every name."""
    counts = pd.DataFrame([dict(parse_counts(c)) for c in count_strings]).fillna(0)
    share = counts.div(pd.Series(totals, dtype=float).clip(lower=1).values, axis=0)
    lift = share / share.mean().where(lambda x: x > 0, 1)
    return share, lift


def pick_categories(share, lift, min_share=0.25):
    """Most over-represented category among those covering at least 25% of the
    event's articles; otherwise CONFLICT. A small category with a high lift
    (e.g. 16% "diplomatic" during a war) shouldn't name the event."""
    out = []
    for i in range(len(share)):
        ok = share.iloc[i] >= min_share
        out.append(lift.iloc[i][ok].idxmax() if ok.any() else "CONFLICT")
    return out


def pick_countries(share, lift, k=5, min_lift=1.5):
    """Countries mentioned at least 1/4 as often as the event's top country AND
    at least 1.5x as often as around other events, ranked by
    share x log(1 + lift): frequent and distinctive. The lift floor keeps
    countries that are in almost all coverage (the US, the UK) out of a name
    unless they really are more present than usual."""
    out = []
    for i in range(len(share)):
        s, l = share.iloc[i], lift.iloc[i]
        cand = s[(s >= 0.25 * s.max()) & (l >= min_lift)] if s.max() > 0 else s.iloc[:0]
        score = (cand * np.log1p(l[cand.index])).sort_values(ascending=False)
        out.append([(c, float(score[c])) for c in score.index[:k]])
    return out


def pick_themes(share, lift, k=6, min_share=0.05):
    out = []
    for i in range(len(share)):
        s, l = share.iloc[i], lift.iloc[i]
        cand = [t for t in s[(s >= min_share) & (l >= 1.2)].index  # more common than usual
                if not t.startswith(GENERIC_THEME_PREFIXES)]
        out.append(list(l[cand].sort_values(ascending=False).index[:k]))
    return out


def name_event(countries, category):
    """Top country, plus the second one if it scores at least 40% as high."""
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

    # per-event counts sit on one row per event (the others are empty)
    first = (news.dropna(subset=["top_countries"]).drop_duplicates("event_date")
             .set_index("event_date").reindex(events["date"]))
    core = first["articles_core"] if "articles_core" in first else first["articles_in_window"]
    totals = core.fillna(0).tolist()
    categories = pick_categories(*shares_and_lift(first["category_counts"].fillna(""), totals))
    countries_all = pick_countries(*shares_and_lift(first["top_countries"].fillna(""), totals))
    themes_all = pick_themes(*shares_and_lift(first["top_themes"].fillna(""), totals))

    rows, news_parts = [], []
    for ev, category, countries, themes in zip(events.itertuples(), categories, countries_all, themes_all):
        n = news[news["event_date"] == ev.date]
        rows.append({
            "date": ev.date.date(), "start_date": ev.start_date.date(), "end_date": ev.end_date.date(),
            "spike_days": ev.spike_days, "geo_tension_index": ev.geo_tension_index, "z_score": ev.z_score,
            "total_events": ev.total_events, "is_spike": True,
            "dominant_category": category,
            "label": name_event(countries[:2], category),
            "category_counts": first["category_counts"].get(ev.date, ""),
            "top_countries": ", ".join(FIPS_NAMES.get(c, c) for c, _ in countries[:5]),
            "top_themes": ", ".join(themes),
            "articles_in_window": int(pd.to_numeric(first["articles_in_window"].get(ev.date), errors="coerce") or 0)
                                  if pd.notna(first["articles_in_window"].get(ev.date)) else 0,
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
