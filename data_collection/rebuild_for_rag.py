"""Rebuild the inputs the RAG Q&A needs, on a laptop, without the NYU cluster.

Why this exists: the original pipeline outputs lived on the NYU Dataproc
cluster, and access ended with the course. This script recreates the parts the
Q&A needs, for the 31 spike dates the original pipeline detected (kept in
dashboard/constants.py), in the same file layout the dashboard used.

What it writes (default: ~/dashboard_data_rebuilt/):
    spike_events.parquet          date, dominant_category (+ provenance flags)
    spike_news/part-0.parquet     same columns as pyspark_pipeline/spike_news_extract.py
    ticker_reaction_by_spike/part-0.parquet
                                  same columns as pyspark_pipeline/event_window.py

Differences from the original outputs (also documented in the README):
  * Geo-Tension Index values, z-scores and thresholds are NOT rebuilt. They
    came from 10 years of data; the spike DATES are reused as detected.
  * News is SAMPLED: --files-per-day GDELT GKG files per day (default 4, one
    every 6 hours) instead of all 96. Same keyword/category filter as
    data_collection/gdelt_download.sh. Articles are ranked by distance from
    the spike day (the original ranked by day -3 first).
  * dominant_category is computed from the sampled international articles on
    the spike day.
  * Weekend/holiday spikes are measured from the next trading day's close
    (the original borrowed the nearest other spike's numbers).
  * Prices: yfinance adjusted closes for the ~50 stocks in SECTOR_MAP only
    (the original used all S&P 500 members; the Q&A only uses these).
    Returns use the same rule as event_window.py: % change vs the spike-day
    close, for calendar days -30..+30.

Run (from the repo root):
    pip install -r requirements.txt
    python -m data_collection.rebuild_for_rag            # ~640 GDELT files, resumable
    python -m rag.build_corpus --data-dir ~/dashboard_data_rebuilt
"""
import argparse
import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from dashboard.constants import SECTOR_MAP, SPIKE_LABELS

NEWS_WINDOW = 3        # days before/after the spike (same as spike_news_extract.py)
PRICE_WINDOW = 30      # days before/after the spike (same as event_window.py)
NEWS_PER_SPIKE = 300   # articles kept per spike
GKG_URL = "http://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"

# Same rules, in the same order, as the awk filter in gdelt_download.sh.
# Note: that script reads GKG column 7 (V2.1 Counts) as "themes" and column 16
# as tone; we keep that exactly so results are comparable.
CATEGORY_RULES = [
    ("CONFLICT", r"MILITARY|WAR|WEAPON|MISSILE|INVASION|AIRSTRIKE"),
    ("ECONOMIC_PRESSURE", r"SANCTION|EMBARGO|TARIFF|EXPORT_CONTROL"),
    ("POLITICAL_INSTABILITY", r"COUP|PROTEST|RIOT"),
    ("DIPLOMATIC_TENSION", r"DIPLOMATIC|EXPULSION"),
    ("DEESCALATION", r"CEASEFIRE"),
    ("INTELLIGENCE_CYBER", r"ESPIONAGE|CYBERATTACK"),
    ("SPECIAL", r"NUCLEAR|BLOCKADE|ANNEXATION"),
]
_COMPILED = [(c, re.compile(p)) for c, p in CATEGORY_RULES]
INTERNATIONAL = re.compile(r"#(?!US)[A-Z]{2}#")


def categorize(themes: str):
    for cat, rx in _COMPILED:
        if rx.search(themes):
            return cat
    return None


def url_to_title(url):
    """Same slug-to-title rule as spike_news_extract.py."""
    try:
        slug = str(url).rstrip("/").split("/")[-1].split(".")[0].split("?")[0]
        title = re.sub(r"[-_]", " ", slug)
        title = re.sub(r"\d+", "", title).strip()
        title = " ".join(w.capitalize() for w in title.split() if len(w) > 2)
        return title if len(title) > 15 else None
    except Exception:
        return None


# ── News ──────────────────────────────────────────────────────────
def spike_dates():
    return sorted(SPIKE_LABELS)


def news_days():
    days = set()
    for s in spike_dates():
        for k in range(-NEWS_WINDOW, NEWS_WINDOW + 1):
            days.add(pd.Timestamp(s) + pd.Timedelta(days=k))
    return sorted(days)


def gkg_timestamps(files_per_day):
    """Evenly spaced 15-minute GKG file timestamps for each needed day."""
    step = 96 // files_per_day  # 96 files per day
    out = []
    for day in news_days():
        for i in range(files_per_day):
            t = day + pd.Timedelta(minutes=15 * step * i)
            out.append(t.strftime("%Y%m%d%H%M%S"))
    return out


def parse_gkg(raw: bytes) -> pd.DataFrame:
    """Unzip one GKG file and keep only geopolitical rows (like the awk filter)."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) < 16:
            continue
        cat = categorize(f[6])
        if cat:
            rows.append((f[0], f[1], f[3], f[4], f[6], f[15], cat))
    return pd.DataFrame(rows, columns=["record_id", "datetime", "domain", "url",
                                       "themes", "tone", "category"])


def fetch_one(ts, cache_dir: Path):
    """Download + filter one file; cache the small filtered result so a
    re-run skips it. Returns 'ok', 'missing' or an error message."""
    import requests

    out = cache_dir / f"{ts}.parquet"
    if out.exists() or (cache_dir / f"{ts}.missing").exists():
        return "cached"
    try:
        r = requests.get(GKG_URL.format(ts=ts), timeout=120)
        if r.status_code == 404:
            (cache_dir / f"{ts}.missing").touch()
            return "missing"
        r.raise_for_status()
        parse_gkg(r.content).to_parquet(out)
        return "ok"
    except Exception as e:  # network hiccup: leave uncached so a re-run retries
        return f"error: {e}"


def download_news(files_per_day, cache_dir: Path, workers: int):
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamps = gkg_timestamps(files_per_day)
    print(f"GDELT: {len(news_days())} days x {files_per_day} files = {len(stamps)} files "
          f"(cached results in {cache_dir})")
    counts = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, ts, cache_dir): ts for ts in stamps}
        for i, fut in enumerate(as_completed(futures), 1):
            status = fut.result()
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "error":
                print(f"  {futures[fut]} {status}")
            if i % 50 == 0 or i == len(stamps):
                print(f"  {i}/{len(stamps)} {counts}")
    if counts.get("error"):
        print(f"{counts['error']} files failed — re-run the same command to retry them.")
    parts = [pd.read_parquet(p) for p in sorted(cache_dir.glob("*.parquet"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_news_outputs(articles: pd.DataFrame):
    articles = articles.copy()
    articles["date"] = pd.to_datetime(articles["datetime"].astype(str).str[:8],
                                      format="%Y%m%d", errors="coerce")
    articles = articles.dropna(subset=["date", "url"]).drop_duplicates("url")

    news_parts, events = [], []
    for s in spike_dates():
        sd = pd.Timestamp(s)
        win = articles[(articles["date"] - sd).dt.days.abs() <= NEWS_WINDOW].copy()
        win["days_from_spike"] = (win["date"] - sd).dt.days

        # dominant category: international articles on the spike day itself
        day = win[(win["days_from_spike"] == 0) & win["themes"].str.contains(INTERNATIONAL)]
        dominant = day["category"].mode().iloc[0] if len(day) else None
        events.append({"date": sd.date(), "dominant_category": dominant,
                       "sampled_articles_spike_day": int(len(day)),
                       "index_rebuilt": False})

        win["title"] = win["url"].map(url_to_title)
        win = win.dropna(subset=["title"])
        win["dist"] = win["days_from_spike"].abs()
        win = win.sort_values(["dist", "datetime"]).head(NEWS_PER_SPIKE)
        news_parts.append(pd.DataFrame({
            "spike_date": sd.date(),
            "dominant_category": dominant,
            "article_date": win["date"].dt.date,
            "domain": win["domain"],
            "url": win["url"],
            "themes": win["themes"],
            "days_from_spike": win["days_from_spike"].astype(int),
            "title": win["title"],
        }))
    return pd.DataFrame(events), pd.concat(news_parts, ignore_index=True)


# ── Prices ────────────────────────────────────────────────────────
def download_prices():
    import yfinance as yf

    tickers = sorted({t for ts in SECTOR_MAP.values() for t in ts})
    start = (pd.Timestamp(spike_dates()[0]) - pd.Timedelta(days=PRICE_WINDOW + 5)).date()
    end = (pd.Timestamp(spike_dates()[-1]) + pd.Timedelta(days=PRICE_WINDOW + 5)).date()
    print(f"yfinance: {len(tickers)} tickers, {start} to {end} (adjusted closes)")
    data = yf.download(tickers, start=str(start), end=str(end),
                       auto_adjust=True, progress=False)["Close"]
    missing = [t for t in tickers if t not in data.columns or data[t].isna().all()]
    if missing:
        print(f"  WARNING: no price data for {missing}")
    return data


def build_reactions(closes: pd.DataFrame) -> pd.DataFrame:
    """Same rule as event_window.py: return vs the spike-day close for every
    trading day within +-30 calendar days of the spike.

    One fix vs the original: if the spike fell on a weekend/holiday, the
    original had no day-0 close and the dashboard borrowed the NEAREST OTHER
    spike's numbers. Here the base is the first trading day on or after the
    spike (recorded in base_date); days_from_spike still counts calendar days
    from the spike date itself."""
    long = (closes.rename_axis("date").reset_index()
            .melt(id_vars="date", var_name="ticker", value_name="close")
            .dropna())
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()

    out = []
    for s in spike_dates():
        sd = pd.Timestamp(s)
        win = long[(long["date"] - sd).dt.days.abs() <= PRICE_WINDOW].copy()
        after = win.loc[win["date"] >= sd, "date"]
        if after.empty:
            continue
        base_date = after.min()
        base = win[win["date"] == base_date].set_index("ticker")["close"]
        win = win[win["ticker"].isin(base.index)]
        win["return_pct"] = (win["close"] / win["ticker"].map(base) - 1) * 100
        out.append(pd.DataFrame({
            "spike_date": sd.date(),
            "base_date": base_date.date(),
            "dominant_category": None,
            "ticker": win["ticker"],
            "days_from_spike": (win["date"] - sd).dt.days.astype(int),
            "avg_return_pct": win["return_pct"],
            "num_obs": 1,
        }))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("~/dashboard_data_rebuilt"))
    ap.add_argument("--files-per-day", type=int, default=4, choices=[1, 2, 4, 8, 12, 24, 48, 96])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    articles = download_news(args.files_per_day, out / "_gkg_cache", args.workers)
    if articles.empty:
        raise SystemExit("No GDELT articles downloaded — check your internet connection and re-run.")
    events, news = build_news_outputs(articles)

    reactions = build_reactions(download_prices())
    dom = dict(zip(events["date"], events["dominant_category"]))
    reactions["dominant_category"] = reactions["spike_date"].map(dom)

    events.to_parquet(out / "spike_events.parquet")
    (out / "spike_news").mkdir(exist_ok=True)
    news.to_parquet(out / "spike_news" / "part-0.parquet")
    (out / "ticker_reaction_by_spike").mkdir(exist_ok=True)
    reactions.to_parquet(out / "ticker_reaction_by_spike" / "part-0.parquet")

    print(f"\nWrote to {out}:")
    print(f"  spike_events: {len(events)} spikes")
    print(f"  spike_news: {len(news)} articles "
          f"({news.groupby('spike_date').size().min()}-{news.groupby('spike_date').size().max()} per spike)")
    print(f"  ticker_reaction_by_spike: {len(reactions)} rows, "
          f"{reactions['spike_date'].nunique()} spikes")
    print(f"\nNext: python -m rag.build_corpus --data-dir {out}")


if __name__ == "__main__":
    main()
