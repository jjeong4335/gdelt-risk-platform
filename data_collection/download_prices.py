"""Download 10 years of daily S&P 500 closes with yfinance (laptop, a few minutes).

Same source as the original project (yfinance). Writes one wide table:
    ~/dashboard_data_rebuilt/sp500_prices.parquet   (rows = dates, columns = tickers)

Notes
  * Ticker list = today's S&P 500 members (from Wikipedia), plus the stocks in
    dashboard/constants.py SECTOR_MAP. Companies that left the index since 2016
    are not included (survivorship bias); the README says so.
  * Adjusted closes (splits and dividends), so a stock split never looks like a crash.
  * Downloads in batches of 50 with a pause in between so Yahoo doesn't block
    the requests. Each batch is cached, so re-running continues where it stopped.

Run (from the repo root):
    python -m data_collection.download_prices
"""
import argparse
import io
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from dashboard.constants import SECTOR_MAP

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BATCH = 50


def sp500_tickers():
    """Today's S&P 500 symbols, in Yahoo's format (BRK.B -> BRK-B)."""
    import requests

    sector_stocks = {t for ts in SECTOR_MAP.values() for t in ts}
    try:
        html = requests.get(WIKI_URL, timeout=30,
                            headers={"User-Agent": "Mozilla/5.0 (research script)"}).text
        table = pd.read_html(io.StringIO(html))[0]
        symbols = {s.replace(".", "-") for s in table["Symbol"].astype(str)}
        print(f"S&P 500 list: {len(symbols)} symbols from Wikipedia")
    except Exception as e:
        print(f"WARNING: couldn't read the S&P 500 list ({e}); using only the {len(sector_stocks)} "
              "sector stocks from dashboard/constants.py")
        symbols = set()
    return sorted(symbols | sector_stocks)


def download(tickers, start, end, cache_dir: Path):
    import yfinance as yf

    cache_dir.mkdir(parents=True, exist_ok=True)
    batches = [tickers[i:i + BATCH] for i in range(0, len(tickers), BATCH)]
    for n, batch in enumerate(batches, 1):
        out = cache_dir / f"batch_{n:03d}.parquet"
        if out.exists():
            print(f"  batch {n}/{len(batches)} already downloaded")
            continue
        for attempt in range(3):
            try:
                data = yf.download(batch, start=start, end=end, auto_adjust=True,
                                   progress=False, threads=True)["Close"]
                if isinstance(data, pd.Series):  # single ticker
                    data = data.to_frame(batch[0])
                data.to_parquet(out)
                got = int(data.notna().any().sum())
                print(f"  batch {n}/{len(batches)}: {got}/{len(batch)} tickers")
                break
            except Exception as e:
                wait = 20 * (attempt + 1)
                print(f"  batch {n} failed ({str(e)[:80]}); retrying in {wait}s")
                time.sleep(wait)
        else:
            print(f"  batch {n} skipped after 3 tries — re-run later to retry it")
        time.sleep(2)
    parts = [pd.read_parquet(p) for p in sorted(cache_dir.glob("batch_*.parquet"))]
    prices = pd.concat(parts, axis=1)
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    return prices.loc[:, ~prices.columns.duplicated()].sort_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("~/dashboard_data_rebuilt"))
    ap.add_argument("--start", default="2015-11-01", help="a little before 2016 for the -30 day window")
    args = ap.parse_args()
    out = args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    tickers = sp500_tickers()
    end = (date.today() + timedelta(days=1)).isoformat()
    prices = download(tickers, args.start, end, out / "_price_cache")
    empty = [t for t in prices.columns if prices[t].isna().all()]
    prices = prices.drop(columns=empty)
    prices.to_parquet(out / "sp500_prices.parquet")
    print(f"\nWrote {out / 'sp500_prices.parquet'}: {prices.shape[1]} tickers, "
          f"{prices.index.min().date()} to {prices.index.max().date()}")
    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        print(f"No data for {len(missing)} tickers (delisted or renamed): {', '.join(missing[:20])}"
              + (" ..." if len(missing) > 20 else ""))
    print("\nNext: python -m data_collection.build_history")


if __name__ == "__main__":
    main()
