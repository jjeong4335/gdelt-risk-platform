"""Cross-event analyses for the Q&A (and anyone reading the data).

Answers three questions from the pipeline outputs, writes analysis.json:
  1. stocks   - which stocks moved most, on average, after the detected spikes
  2. vix_lead - did the VIX or the Geo-Tension score move first around spikes
  3. volatile - were the biggest movers similar from one event to the next

Inputs (in ~/dashboard_data_rebuilt/): ticker_reaction_by_spike/, spike_events.parquet,
geo_tension_index.parquet, vix.parquet (from download_prices.py; step 2 is skipped without it).

Run (from the repo root; build_events.py also runs it at the end):
    python -m data_collection.build_analysis
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.constants import SECTOR_MAP

TOP_N = 10          # stocks listed per direction
MOVERS_N = 20       # "biggest movers" per event
LAG_WINDOW = 10     # days either side of the spike to look for the VIX peak


def stock_ranking(reac: pd.DataFrame, n_events: int) -> dict:
    sector = {t: s for s, ts in SECTOR_MAP.items() for t in ts}
    out = {}
    for h in (5, 30):
        d = reac[reac["days_from_spike"] == h]
        g = d.groupby("ticker")["avg_return_pct"]
        stats = pd.DataFrame({"avg": g.mean(), "events": g.size(), "up_share": g.apply(lambda x: (x > 0).mean())})
        stats = stats[stats["events"] >= max(5, n_events // 2)]  # skip stocks with few events
        rows = lambda df: [{"ticker": t, "sector": sector.get(t), "avg_return_pct": round(r.avg, 2),
                            "rose_in": f"{int(round(r.up_share * r.events))} of {int(r.events)} events"}
                           for t, r in df.iterrows()]
        out[f"day{h}"] = {"best": rows(stats.nlargest(TOP_N, "avg")),
                          "worst": rows(stats.nsmallest(TOP_N, "avg")),
                          "stocks_compared": int(len(stats))}
    return out


def vix_lead(events: pd.DataFrame, tension: pd.DataFrame, vix: pd.DataFrame) -> dict:
    t = tension.set_index("date")["z_score"]
    v = vix.set_index("date")["vix"]
    lags = []
    for d in events["date"]:
        w = v[(v.index >= d - pd.Timedelta(days=LAG_WINDOW)) & (v.index <= d + pd.Timedelta(days=LAG_WINDOW))]
        if len(w):
            lags.append({"event": str(d.date()), "vix_peak": str(w.idxmax().date()),
                         "vix_peak_minus_tension_peak_days": int((w.idxmax() - d).days)})
    days = [x["vix_peak_minus_tension_peak_days"] for x in lags]
    # Whole-period check: correlation of tension(t) with the VIX daily % change k days later.
    both = pd.concat([t, v.pct_change() * 100], axis=1, keys=["t", "dv"], sort=True).dropna()
    corr = {k: round(float(both["t"].corr(both["dv"].shift(-k))), 3) for k in range(-5, 6)}
    return {"per_event": lags,
            "median_days": float(np.median(days)) if days else None,
            "tension_first": sum(x > 0 for x in days), "vix_first": sum(x < 0 for x in days),
            "same_day": sum(x == 0 for x in days), "events": len(days),
            "corr_tension_vs_vix_change_k_days_later": corr}


def volatile_overlap(reac: pd.DataFrame) -> dict:
    d = reac[reac["days_from_spike"] == 5].assign(abs_ret=lambda x: x["avg_return_pct"].abs())
    tops = {str(e): set(g.nlargest(MOVERS_N, "abs_ret")["ticker"]) for e, g in d.groupby("spike_date")}
    universe = d["ticker"].nunique()
    pairs = [len(a & b) / len(a | b) for a, b in combinations(tops.values(), 2)]
    counts = pd.Series([t for s in tops.values() for t in s]).value_counts()
    # Overlap two random sets of MOVERS_N stocks would have, for comparison.
    p = MOVERS_N / universe
    random_jaccard = (MOVERS_N * p) / (2 * MOVERS_N - MOVERS_N * p)
    return {"movers_per_event": MOVERS_N, "stocks": int(universe), "events": len(tops),
            "mean_jaccard": round(float(np.mean(pairs)), 3) if pairs else None,
            "random_jaccard": round(random_jaccard, 3),
            "most_repeated": [{"ticker": t, "events": int(c)} for t, c in counts.head(12).items()]}


def build(d: Path) -> dict:
    reac = pd.read_parquet(d / "ticker_reaction_by_spike")
    events = pd.read_parquet(d / "spike_events.parquet")
    events["date"] = pd.to_datetime(events["date"])
    out = {"stocks": stock_ranking(reac, len(events)), "volatile": volatile_overlap(reac)}
    tension = pd.read_parquet(d / "geo_tension_index.parquet")
    tension["date"] = pd.to_datetime(tension["date"])
    if (d / "vix.parquet").exists() and "z_score" in tension:
        vix = pd.read_parquet(d / "vix.parquet")
        vix["date"] = pd.to_datetime(vix["date"])
        out["vix_lead"] = vix_lead(events, tension, vix)
    (d / "analysis.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("~/dashboard_data_rebuilt"))
    d = ap.parse_args().data_dir.expanduser()
    out = build(d)
    print(f"Wrote {d / 'analysis.json'}")
    s = out["stocks"]["day5"]
    print("  best Day +5 on average:", ", ".join(f"{x['ticker']} {x['avg_return_pct']:+.2f}%" for x in s["best"][:5]))
    v = out["volatile"]
    print(f"  biggest movers overlap between events: {v['mean_jaccard']} (random sets: {v['random_jaccard']})")
    if "vix_lead" in out:
        x = out["vix_lead"]
        print(f"  VIX peak vs tension peak: median {x['median_days']:+.1f} days "
              f"(tension first {x['tension_first']}, VIX first {x['vix_first']}, same day {x['same_day']})")
    else:
        print("  VIX step skipped: run python -m data_collection.download_prices (it now saves vix.parquet).")


if __name__ == "__main__":
    main()
