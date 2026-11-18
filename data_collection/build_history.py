"""STEP 2 of the data-driven rebuild: daily index + automatic spike detection.

Input:  ~/dashboard_data_rebuilt/gdelt_daily.csv   (from bigquery_daily_index.sql)

What it does
  1. Same index formula as the original Spark job (geo_tension_index.py):
         raw   = avg_negative_tone * ln(article_count + 1)
         index = raw scaled to 0..10 with the 1st/99th percentiles (for display)
  2. Detects spikes on the RAW score:  raw > that year's mean + 3 * std.
     (The original detected on the 0-10 index. Because the index is capped
     at 10, the threshold rose above 10 in volatile years and no spike could
     ever be detected there.)
  3. Groups spike days that are at most 3 days apart into one EVENT; the
     event's date is its peak day. No dates or names are typed in by hand.
  4. Writes the next BigQuery query (event_news.sql) with those event windows
     filled in, so the news for each event is also pulled from the data.

Outputs (in the data folder):
  geo_tension_index.parquet   daily series for the dashboard
  events.csv                  detected events (start, end, peak, z-score)
  event_news.sql              STEP 3 query to paste into BigQuery

Run (from the repo root):
    python -m data_collection.build_history
"""
import argparse
import math
from pathlib import Path

import pandas as pd

GROUP_GAP_DAYS = 3      # spike days this close together are one event
NEWS_WINDOW_DAYS = 3    # news collected from peak - 3 to peak + 3 days
ARTICLES_PER_EVENT = 150
SQL_TEMPLATE = Path(__file__).parent / "bigquery_event_news.sql"


def build_index(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date")
    # The query runs mid-day, so the last day is usually incomplete. Drop it so
    # "today's index" isn't artificially low.
    recent = df["total_events"].iloc[-31:-1].median()
    if len(df) > 31 and df["total_events"].iloc[-1] < 0.5 * recent:
        print(f"Dropping incomplete last day {df['date'].iloc[-1].date()} "
              f"({df['total_events'].iloc[-1]} articles vs ~{recent:.0f} usual)")
        df = df.iloc[:-1]

    df["geo_tension_raw"] = df["avg_negative_tone"] * (df["total_events"] + 1).map(math.log)
    p1, p99 = df["geo_tension_raw"].quantile([0.01, 0.99])
    df["geo_tension_index"] = ((df["geo_tension_raw"] - p1) / (p99 - p1) * 10).clip(0, 10)

    df["year"] = df["date"].dt.year
    g = df.groupby("year")["geo_tension_raw"]
    df["raw_mean"], df["raw_std"] = g.transform("mean"), g.transform("std")
    df["z_score"] = (df["geo_tension_raw"] - df["raw_mean"]) / df["raw_std"]
    df["is_spike"] = df["z_score"] > 3
    return df


def group_events(df: pd.DataFrame) -> pd.DataFrame:
    """Turn spike days into events: consecutive spike days (<= 3 days apart)
    form one event, dated by its peak (highest raw score) day."""
    spikes = df[df["is_spike"]].sort_values("date")
    events, current = [], []
    for _, row in spikes.iterrows():
        if current and (row["date"] - current[-1]["date"]).days > GROUP_GAP_DAYS:
            events.append(current)
            current = []
        current.append(row)
    if current:
        events.append(current)

    rows = []
    for days in events:
        peak = max(days, key=lambda r: r["geo_tension_raw"])
        rows.append({
            "date": peak["date"].date(),
            "start_date": days[0]["date"].date(),
            "end_date": days[-1]["date"].date(),
            "spike_days": len(days),
            "geo_tension_index": round(float(peak["geo_tension_index"]), 2),
            "z_score": round(float(peak["z_score"]), 2),
            "total_events": int(peak["total_events"]),
            "dominant_category": peak["dominant_category"],
        })
    return pd.DataFrame(rows)


def write_news_sql(events: pd.DataFrame, path: Path):
    """Fill the STEP 3 query template with the detected event windows."""
    structs, days = [], set()
    for e in events.itertuples():
        peak = pd.Timestamp(e.date)
        start = peak - pd.Timedelta(days=NEWS_WINDOW_DAYS)
        end = peak + pd.Timedelta(days=NEWS_WINDOW_DAYS)
        structs.append(f"    STRUCT(DATE '{peak:%Y-%m-%d}' AS event_date, "
                       f"DATE '{start:%Y-%m-%d}' AS start_day, DATE '{end:%Y-%m-%d}' AS end_day)")
        # one extra partition on each side: partitions follow ingestion time
        for k in range(-NEWS_WINDOW_DAYS - 1, NEWS_WINDOW_DAYS + 2):
            days.add(peak + pd.Timedelta(days=k))
    partitions = ",\n    ".join(f"TIMESTAMP('{d:%Y-%m-%d}')" for d in sorted(days))
    sql = (SQL_TEMPLATE.read_text()
           .replace("{{EVENTS}}", ",\n".join(structs))
           .replace("{{PARTITIONS}}", partitions)
           .replace("{{ARTICLES_PER_EVENT}}", str(ARTICLES_PER_EVENT)))
    path.write_text(sql)
    return len(days)


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

    out = df[["date", "geo_tension_index", "geo_tension_raw", "z_score", "total_events"]].copy()
    out["date"] = out["date"].dt.date
    out.to_parquet(d / "geo_tension_index.parquet")

    events = group_events(df)
    events.to_csv(d / "events.csv", index=False)
    per_year = df[df["is_spike"]].groupby("year").size().to_dict()
    print(f"\nDetected {int(df['is_spike'].sum())} spike days (raw score > yearly mean + 3 sd), "
          f"grouped into {len(events)} events.")
    print(f"  spike days per year: {per_year}")

    n_days = write_news_sql(events, d / "event_news.sql")
    print(f"\nWrote {d / 'event_news.sql'} (reads {n_days} days of GDELT).")
    print("Next: paste it into BigQuery (check the GB estimate first), save the result as "
          f"{d / 'event_news.csv'}, then run:  python -m data_collection.build_events")


if __name__ == "__main__":
    main()
