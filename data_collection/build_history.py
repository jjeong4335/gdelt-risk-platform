"""Rebuild the 10-year history the dashboard needs, with pandas instead of Spark.

Inputs (in ~/dashboard_data_rebuilt/):
    gdelt_daily.csv        from data_collection/bigquery_daily_index.sql (one row per day)
    sp500_prices.parquet   from data_collection/download_prices.py

Steps, mirroring the original Spark jobs:
  1. geo_tension_index.py
       raw   = avg_negative_tone * ln(article_count + 1)
       index = raw scaled to 0..10 using the 1st and 99th percentile of all days
       spike = index > that year's mean + 3 * that year's std
  2. Spike check: are the original 31 spike dates detected again? (printed and
     saved to spike_check.csv — a good sanity check of the rebuild)
  3. event_window.py
       for each spike and every stock: % return vs the spike-day close,
       calendar days -30..+30, then a per-ticker summary (ticker_summary).

Outputs (same names the dashboard and RAG code already read):
    geo_tension_index.parquet, spike_events.parquet,
    ticker_reaction_by_spike/part-0.parquet, ticker_summary.parquet, spike_check.csv

The spike list stays the original 31 dates (they have labels and news), with
the new index, z-score and threshold attached.

Run (from the repo root):
    python -m data_collection.build_history
"""
import argparse
import math
from pathlib import Path

import pandas as pd

from dashboard.constants import SPIKE_LABELS
from data_collection.rebuild_for_rag import build_reactions


def build_index(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date")
    # The query runs mid-day, so the last day is usually incomplete (a few
    # hundred articles instead of thousands). Drop it so "today's index"
    # isn't artificially low.
    recent = df["total_events"].iloc[-31:-1].median()
    if len(df) > 31 and df["total_events"].iloc[-1] < 0.5 * recent:
        print(f"Dropping incomplete last day {df['date'].iloc[-1].date()} "
              f"({df['total_events'].iloc[-1]} articles vs ~{recent:.0f} usual)")
        df = df.iloc[:-1]
    df["geo_tension_raw"] = df["avg_negative_tone"] * (df["total_events"] + 1).map(math.log)
    p1, p99 = df["geo_tension_raw"].quantile([0.01, 0.99])
    print(f"Raw tension percentiles: p1={p1:.3f}, p99={p99:.3f}")
    df["geo_tension_index"] = ((df["geo_tension_raw"] - p1) / (p99 - p1) * 10).clip(0, 10)

    df["year"] = df["date"].dt.year
    stats = df.groupby("year")["geo_tension_index"].agg(year_mean="mean", year_std="std")
    df = df.join(stats, on="year")
    df["threshold"] = df["year_mean"] + 3 * df["year_std"]
    df["z_score"] = (df["geo_tension_index"] - df["year_mean"]) / df["year_std"]
    df["is_spike"] = df["geo_tension_index"] > df["threshold"]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("~/dashboard_data_rebuilt"))
    args = ap.parse_args()
    d = args.data_dir.expanduser()

    csv = d / "gdelt_daily.csv"
    if not csv.exists():
        raise SystemExit(f"{csv} not found. Run data_collection/bigquery_daily_index.sql in the "
                         "BigQuery console and save the result there as CSV.")
    df = build_index(pd.read_csv(csv))
    print(f"Daily index: {len(df)} days, {df['date'].min().date()} to {df['date'].max().date()}")

    # 1) the 10-year series, same columns as the original output
    out = df[["date", "geo_tension_index", "total_events", "dominant_category"]].copy()
    out["date"] = out["date"].dt.date
    out.to_parquet(d / "geo_tension_index.parquet")

    # 2) spike check against the original 31 dates
    detected = set(df.loc[df["is_spike"], "date"].dt.strftime("%Y-%m-%d"))
    original = set(SPIKE_LABELS)
    both = original & detected
    check = pd.DataFrame(
        [{"date": x, "in_original_31": x in original, "detected_now": x in detected,
          "label": SPIKE_LABELS.get(x, "")} for x in sorted(original | detected)])
    check.to_csv(d / "spike_check.csv", index=False)
    print(f"\nSpike check: {len(detected)} days detected now; "
          f"{len(both)}/{len(original)} of the original 31 spike dates detected again.")
    z_orig = df[df["date"].dt.strftime("%Y-%m-%d").isin(original)]["z_score"]
    print(f"  z-score of the original dates: >=3: {(z_orig >= 3).sum()}, >=2: {(z_orig >= 2).sum()}, "
          f"<1: {(z_orig < 1).sum()} (of {len(z_orig)})")
    capped = df.groupby("year")["threshold"].first()
    capped = capped[capped > 10]
    if len(capped):
        print("  years where the threshold is above the 0-10 cap (no spike possible): "
              + ", ".join(f"{y} ({t:.2f})" for y, t in capped.items()))
    if original - detected:
        print("  original dates NOT detected now:", ", ".join(sorted(original - detected)))
    if detected - original:
        print("  new dates detected (not in the original 31):", ", ".join(sorted(detected - original)))

    # spike_events: the original 31 dates with the new numbers attached
    by_date = df.set_index(df["date"].dt.strftime("%Y-%m-%d"))
    rows = []
    for x in sorted(original):
        r = by_date.loc[x] if x in by_date.index else None
        rows.append({
            "date": pd.Timestamp(x).date(),
            "geo_tension_index": None if r is None else float(r["geo_tension_index"]),
            "total_events": None if r is None else int(r["total_events"]),
            "threshold": None if r is None else float(r["threshold"]),
            "z_score": None if r is None else float(r["z_score"]),
            "is_spike": True,
            "detected_again": x in detected,
            "dominant_category": None if r is None else r["dominant_category"],
        })
    pd.DataFrame(rows).to_parquet(d / "spike_events.parquet")

    # 3) stock reactions for every S&P 500 stock + per-ticker summary
    prices_path = d / "sp500_prices.parquet"
    if not prices_path.exists():
        print(f"\n{prices_path} not found — skipped stock reactions "
              "(run python -m data_collection.download_prices).")
        return
    closes = pd.read_parquet(prices_path)
    reactions = build_reactions(closes)
    dom = {r["date"]: r["dominant_category"] for r in rows}
    reactions["dominant_category"] = reactions["spike_date"].map(dom)
    (d / "ticker_reaction_by_spike").mkdir(exist_ok=True)
    reactions.to_parquet(d / "ticker_reaction_by_spike" / "part-0.parquet")

    g = reactions.groupby("ticker")
    summary = pd.DataFrame({
        "avg_return_5d": reactions[reactions["days_from_spike"] == 5].groupby("ticker")["avg_return_pct"].mean(),
        "avg_return_30d": reactions[reactions["days_from_spike"] == 30].groupby("ticker")["avg_return_pct"].mean(),
        "worst_drawdown": g["avg_return_pct"].min(),
        "num_events": g["spike_date"].nunique(),
    }).reset_index()
    summary.to_parquet(d / "ticker_summary.parquet")
    print(f"\nStock reactions: {reactions['ticker'].nunique()} tickers x "
          f"{reactions['spike_date'].nunique()} spikes = {len(reactions):,} rows")
    print("\nNext: python -m rag.build_corpus --data-dir", d)


if __name__ == "__main__":
    main()
