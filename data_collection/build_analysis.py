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
LAG_WINDOW = 5      # trading days either side of the news peak to look for the VIX jump


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
    """For each event, the trading day on which the VIX jumped the most (largest
    daily % rise) within LAG_WINDOW trading days of the news peak, counted in
    trading days from the first trading day on/after the peak (day 0 = the
    market's first chance to react, so weekend news isn't counted as a lag).
    A jump before day 0 means the VIX moved before the news peaked."""
    t = tension.set_index("date")["z_score"]
    v = vix.set_index("date")["vix"].sort_index()
    jump = v.pct_change() * 100
    days_idx = v.index
    lags = []
    for d in events["date"]:
        pos0 = days_idx.searchsorted(d)  # first trading day on/after the news peak
        if pos0 >= len(days_idx):
            continue
        lo, hi = max(1, pos0 - LAG_WINDOW), min(len(days_idx) - 1, pos0 + LAG_WINDOW)
        w = jump.iloc[lo:hi + 1]
        k = int(np.argmax(w.values))
        lags.append({"event": str(d.date()), "vix_biggest_jump_day": str(w.index[k].date()),
                     "jump_pct": round(float(w.iloc[k]), 1),
                     "trading_days_after_news_peak": int(lo + k - pos0)})
    days = [x["trading_days_after_news_peak"] for x in lags]
    # Whole-period check: correlation of tension(t) with the VIX daily % change k days later.
    both = pd.concat([t, jump], axis=1, keys=["t", "dv"], sort=True).dropna()
    corr = {k: round(float(both["t"].corr(both["dv"].shift(-k))), 3) for k in range(-5, 6)}
    # Co-movement, whichever moves first: VIX % change from the close before the
    # news peak's trading day (day -1) to the close after it (day +1), compared
    # with the same 2-day change on every other day, and how often random dates
    # match the events' average (permutation test).
    two_day = (v.shift(-1) / v.shift(1) - 1) * 100
    ev_pos = sorted({int(days_idx.searchsorted(d)) for d in events["date"]} - {len(days_idx)})
    ev_vals = two_day.iloc[ev_pos].dropna()
    near = set(p for e in ev_pos for p in range(e - 3, e + 4))
    normal = two_day.iloc[[p for p in range(len(two_day)) if p not in near]].dropna()
    rng = np.random.default_rng(0)
    sims = rng.choice(normal.values, size=(10000, len(ev_vals))).mean(axis=1)
    comove = {"event_mean_pct": round(float(ev_vals.mean()), 2), "normal_mean_pct": round(float(normal.mean()), 2),
              "event_median_pct": round(float(ev_vals.median()), 2),
              "events_vix_up": int((ev_vals > 0).sum()), "events": int(len(ev_vals)),
              "normal_share_up": round(float((normal > 0).mean()), 3),
              "p_value": round(float((sims >= ev_vals.mean()).mean()), 4)}
    return {"comove": comove, "per_event": lags, "window_trading_days": LAG_WINDOW,
            "median_days": float(np.median(days)) if days else None,
            "vix_before": sum(x < 0 for x in days), "vix_same_day": sum(x == 0 for x in days),
            "vix_after": sum(x > 0 for x in days), "events": len(days),
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


EVENT_GROUPS = {  # dominant_category -> group used for the sector tests
    "CONFLICT": "war & conflict", "SPECIAL": "war & conflict", "ECONOMIC_PRESSURE": "war & conflict",
    "TERRORISM": "terror attacks",
}
HOLD_DAYS = 5   # trading days after the news peak


def sector_tests(events: pd.DataFrame, prices: pd.DataFrame, n_sims: int = 10000) -> dict:
    """Did sectors move differently after spikes than on normal days?

    Excess return = sector return - market return, where a sector is the
    equal-weighted average of its S&P 500 members (dashboard/constants.py) and
    the market is the equal-weighted average of all stocks we have, so the
    2016-2026 bull market doesn't count as an effect. Window: close of the
    trading day before the news peak's first trading day (day 0) to the close
    of day +HOLD_DAYS. Compared with the same window on every other start day
    (at least 7 trading days from any event), with a random-date test
    (two-sided p-value), for all events and for each event group."""
    prices = prices.sort_index()
    ret = prices.pct_change()
    market = ret.mean(axis=1)
    sector_ret = pd.DataFrame({s: ret[[t for t in ts if t in ret]].mean(axis=1)
                               for s, ts in SECTOR_MAP.items() if any(t in ret for t in ts)})
    excess = sector_ret.sub(market, axis=0)
    # cumulative excess over days 0..HOLD_DAYS, indexed by the day-0 position
    win = (np.log1p(excess.fillna(0)).rolling(HOLD_DAYS + 1).sum().shift(-HOLD_DAYS)).pipe(np.expm1) * 100
    idx = prices.index
    pos = {str(d.date()): int(idx.searchsorted(d)) for d in events["date"]}
    pos = {k: v for k, v in pos.items() if v < len(idx) - HOLD_DAYS}
    near = {p for v in pos.values() for p in range(v - 7, v + 8)}
    normal_pos = [p for p in range(1, len(idx) - HOLD_DAYS) if p not in near]
    groups = {"all events": list(pos)}
    cat = dict(zip(events["date"].dt.date.astype(str), events["dominant_category"]))
    for d in pos:
        groups.setdefault(EVENT_GROUPS.get(cat.get(d), "other"), []).append(d)
    rng = np.random.default_rng(0)
    out = {}
    for g, dates in groups.items():
        if len(dates) < 3:
            continue
        res = {}
        for sec in win.columns:
            normal = win[sec].iloc[normal_pos].dropna().values
            ev = win[sec].iloc[[pos[d] for d in dates]].dropna().values
            if len(ev) < 3 or len(normal) < 100:
                continue
            base = normal.mean()
            sims = rng.choice(normal, size=(n_sims, len(ev))).mean(axis=1)
            p = float((np.abs(sims - base) >= abs(ev.mean() - base)).mean())
            res[sec] = {"event_mean_pct": round(float(ev.mean()), 2), "normal_mean_pct": round(float(base), 2),
                        "up_share": round(float((ev > 0).mean()), 2), "events": int(len(ev)), "p_value": round(p, 4)}
        out[g] = res
    return {"hold_days": HOLD_DAYS, "groups": {g: len(v) for g, v in groups.items()}, "tests": out}


def timing(events: pd.DataFrame, tension: pd.DataFrame, prices: pd.DataFrame, vix, days=range(-3, 6)) -> dict:
    """Day-by-day average around war/conflict events (day 0 = first trading day
    on/after the news peak): tension z-score, VIX daily % change, and daily
    excess return (vs the average stock) of Defense and Airlines. Shows which
    one reacted first and on which day most of each move happened."""
    war = events[events["dominant_category"].map(EVENT_GROUPS).eq("war & conflict")]
    idx = prices.sort_index().index
    ret = prices.sort_index().pct_change()
    market = ret.mean(axis=1)
    series = {s: (ret[[t for t in SECTOR_MAP[s] if t in ret]].mean(axis=1) - market) * 100
              for s in ("Defense", "Airlines")}
    if vix is not None:
        series["VIX"] = vix.set_index("date")["vix"].reindex(idx).pct_change() * 100
    tz = tension.set_index("date")["z_score"]
    out = {s: {} for s in ["Tension (z)"] + list(series)}
    for k in days:
        vals = {s: [] for s in out}
        for d in war["date"]:
            p0 = int(idx.searchsorted(d))
            if not (0 <= p0 + k < len(idx)):
                continue
            day = idx[p0 + k]
            vals["Tension (z)"].append(tz.get(day, np.nan))
            for s, ser in series.items():
                vals[s].append(ser.iloc[p0 + k])
        for s in out:
            v = pd.Series(vals[s], dtype=float).dropna()
            out[s][k] = round(float(v.mean()), 2) if len(v) else None
    return {"events": int(len(war)), "by_day": out}


def build(d: Path) -> dict:
    reac = pd.read_parquet(d / "ticker_reaction_by_spike")
    events = pd.read_parquet(d / "spike_events.parquet")
    events["date"] = pd.to_datetime(events["date"])
    out = {"stocks": stock_ranking(reac, len(events)), "volatile": volatile_overlap(reac)}
    vix = None
    if (d / "vix.parquet").exists():
        vix = pd.read_parquet(d / "vix.parquet")
        vix["date"] = pd.to_datetime(vix["date"])
    tension = pd.read_parquet(d / "geo_tension_index.parquet")
    tension["date"] = pd.to_datetime(tension["date"])
    if (d / "sp500_prices.parquet").exists():
        prices = pd.read_parquet(d / "sp500_prices.parquet")
        out["sectors"] = sector_tests(events, prices)
        if "z_score" in tension:
            out["timing"] = timing(events, tension, prices, vix)
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
    if "timing" in out:
        t = out["timing"]
        print(f"  Day by day around the {t['events']} war/conflict events (day 0 = first trading day after the news peak):")
        ks = list(next(iter(t["by_day"].values())).keys())
        print("    " + " " * 13 + "".join(f"{('day ' + str(k)) if k else 'day 0':>9}" for k in ks))
        for name, row in t["by_day"].items():
            print(f"    {name:<13}" + "".join(f"{(v if v is not None else float('nan')):>+9.2f}" for v in row.values()))
        print("    (Tension = z-score; VIX = daily % change; Defense/Airlines = daily % return minus the average stock)")
    for g, res in out.get("sectors", {}).get("tests", {}).items():
        line = ", ".join(f"{sec} {r['event_mean_pct']:+.2f}%{'*' if r['p_value'] < 0.05 else ''}"
                         for sec, r in res.items())
        print(f"  {g} ({out['sectors']['groups'][g]}): sector excess return, day 0 to +{HOLD_DAYS}: {line}")
    if "sectors" in out:
        print("    (* = p < 0.05 vs normal days)")
    if "vix_lead" in out:
        x = out["vix_lead"]
        c = x["comove"]
        print(f"  VIX change day -1 to +1 around spikes: {c['event_mean_pct']:+.2f}% vs {c['normal_mean_pct']:+.2f}% "
              f"on normal days; VIX rose in {c['events_vix_up']}/{c['events']} events; p = {c['p_value']}")
        print(f"  Biggest VIX jump vs news peak: median {x['median_days']:+.1f} trading days "
              f"(VIX before {x['vix_before']}, first trading day {x['vix_same_day']}, after {x['vix_after']})")
    else:
        print("  VIX step skipped: run python -m data_collection.download_prices (it now saves vix.parquet).")


if __name__ == "__main__":
    main()
