"""Build the text corpus for the Q&A from the pipeline outputs.

Reads the files written by data_collection/build_events.py (in GEO_DATA_DIR,
default ~/dashboard_data_rebuilt) and writes one JSON line per "document" to
rag/data/corpus.jsonl.

Three kinds of documents:
  * one EVENT document per detected spike  -> "what happened on date X?",
    "how big was the spike?", "which sector fell most?"
  * one EVENT x SECTOR document            -> "how did airlines react to X?"
  * comparison summaries (per sector + one over all events) -> questions
    that compare ALL events, which top-k retrieval can't do on its own.
Event and sector documents stay under the embedding model's 256-token limit.

Run (from the repo root):
    python -m rag.build_corpus
    python -m rag.build_corpus --data-dir /path/to/data
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from dashboard.constants import SECTOR_MAP
from rag import config

# The pipeline outputs this script needs, relative to the data folder.
REQUIRED_OUTPUTS = {
    "spike_events": "spike_events.parquet",
    "spike_news": "spike_news",
    "reactions": "ticker_reaction_by_spike",
}
# The three horizons the Q&A talks about (calendar days after the spike).
HORIZONS = [1, 5, 30]
# Kept small so an event document stays under the model's 256-token limit.
HEADLINES_PER_EVENT = 3


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
        out[key] = pd.read_parquet(data_dir / name) if (data_dir / name).exists() else None
    for key, col in [("spike_events", "date"), ("spike_news", "spike_date"), ("reactions", "spike_date")]:
        if out[key] is not None:
            out[key][col] = _to_date_str(out[key][col])
    a = data_dir / "analysis.json"
    out["analysis"] = json.loads(a.read_text()) if a.exists() else None
    return out


def clean_headlines(news: pd.DataFrame) -> pd.DataFrame:
    """Same quality filter the dashboard uses for URL-derived titles."""
    t = news["title"].fillna("")
    mask = (
        (t.str.len() > 20)
        & ~t.str.contains(r"%|#|\.com|\.org|\.net", regex=True)  # % = URL-encoded non-Latin title
        & ~t.str.contains(r"[A-Fa-f0-9]{6,}", regex=True)
        & t.str.contains(" ")
        & (t.str.replace(r"[^a-zA-Z]", "", regex=True).str.len() > 15)
    )
    return news[mask]


def sector_reactions(reactions: pd.DataFrame, date: str):
    """Average return per sector at Day +1/+5/+30.

    Returns (table, reaction_date, base_date). Like the dashboard:
      * if the spike day was not a trading day, use the nearest spike date
        that has price data;
      * if Day +N was a weekend/holiday, use the nearest trading day and
        record which day was actually used.
    """
    available = reactions["spike_date"].unique()
    if len(available) == 0:
        return {}, None, None
    if date in set(available):
        reaction_date = date
    else:
        d = pd.Timestamp(date)
        reaction_date = min(available, key=lambda x: abs((pd.Timestamp(x) - d).days))
    rows = reactions[reactions["spike_date"] == reaction_date].copy()

    ticker_to_sector = {t: s for s, tickers in SECTOR_MAP.items() for t in tickers}
    rows["sector"] = rows["ticker"].map(ticker_to_sector)
    rows = rows.dropna(subset=["sector"])
    base_date = (str(pd.Timestamp(rows["base_date"].iloc[0]).date())
                 if "base_date" in rows and len(rows) else None)
    days = sorted(d for d in rows["days_from_spike"].unique() if d > 0)
    if not days:
        return {}, reaction_date, base_date

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
    return table, reaction_date, base_date


def fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


def horizon_text(cell: dict, horizon: int) -> str:
    s = f"Day +{horizon}: {fmt_pct(cell['return_pct'])}"
    if cell["day_used"] != horizon:
        s += f" (nearest trading day, +{cell['day_used']})"
    return s


def build_documents(data: dict) -> list:
    se = data["spike_events"]
    missing = [c for c in ("label", "z_score", "start_date", "end_date", "top_countries") if c not in se]
    if missing:
        sys.exit(f"spike_events.parquet lacks {missing}: run python -m data_collection.build_events first.")
    news_all = data["spike_news"]
    docs = []

    for _, ev in data["spike_events"].sort_values("date").iterrows():
        date = ev["date"]
        news = news_all[news_all["spike_date"] == date]
        label = ev["label"]
        headlines = (clean_headlines(news)
                     .assign(dist=lambda d: d["days_from_spike"].abs())
                     .sort_values("dist")
                     .drop_duplicates("title")
                     .head(HEADLINES_PER_EVENT))
        urls = headlines["url"].tolist()
        source_url = urls[0] if urls else ""
        countries = [c for c in str(ev["top_countries"] or "").split(", ") if c]
        z_score = float(ev["z_score"])
        sectors, reaction_date, base_date = sector_reactions(data["reactions"], date)

        # ── EVENT document ──
        parts = [f"Event: {label}. Spike date: {date}."]
        start, end = str(pd.Timestamp(ev["start_date"]).date()), str(pd.Timestamp(ev["end_date"]).date())
        if start != end:
            parts.append(f"Tension stayed elevated from {start} to {end} "
                         f"({int(ev['spike_days'])} spike days).")
        parts.append(
            f"Geo-Tension Index: {ev['geo_tension_index']:.2f} out of 10; the raw tension score was "
            f"{z_score:.2f} standard deviations above the {date[:4]} average (spike rule: above 3).")
        if pd.notna(ev.get("dominant_category")):
            parts.append(f"Dominant news category: {ev['dominant_category']}.")
        if pd.notna(ev.get("total_events")):
            parts.append(f"International articles on the peak day: {int(ev['total_events']):,}.")
        if countries:
            parts.append("Countries that stood out in coverage (mentioned often, and far more than "
                         "around other spikes), most distinctive first: " + ", ".join(countries) + ".")
        if pd.notna(ev.get("most_mentioned")) and ev["most_mentioned"]:
            parts.append("Countries in the most articles (plain count, including the United States): "
                         + ev["most_mentioned"] + ".")
        if pd.notna(ev.get("top_themes")) and ev["top_themes"]:
            parts.append("Top GDELT news themes: " + ev["top_themes"].replace("_", " ").lower() + ".")
        if len(headlines):
            parts.append("Sample headlines (taken from article URLs): " + "; ".join(
                f"\"{r.title}\" ({r.domain})"
                for r in headlines.itertuples()) + ".")
        if sectors:
            day5 = sorted(((s, v[5]) for s, v in sectors.items() if 5 in v),
                          key=lambda x: x[1]["return_pct"])
            parts.append("Sector average returns at Day +5 after the spike: " + ", ".join(
                f"{s} {fmt_pct(c['return_pct'])}" for s, c in day5) + ".")
            parts.append(f"Sector that fell the most (weakest) at Day +5: {day5[0][0]}. "
                         f"Sector that rose the most (strongest) at Day +5: {day5[-1][0]}.")
            if base_date and base_date != date:
                parts.append(f"(Spike was not a trading day; returns use the {base_date} close as the base.)")
            elif reaction_date != date:
                parts.append(f"(No price data for this date; numbers are from the nearest spike "
                             f"date with price data, {reaction_date}, as in the dashboard.)")
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
        base_label = (f"{base_date} (first trading day after the spike)"
                      if base_date and base_date != date else "spike-day")
        for sector, cells in sectors.items():
            tickers = ", ".join(SECTOR_MAP[sector])
            moves = "; ".join(horizon_text(cells[h], h) for h in HORIZONS if h in cells)
            text = (
                f"Sector reaction: {sector} stocks after the spike on {date} ({label}). "
                f"Average return of S&P 500 members ({tickers}) measured from the "
                f"{base_label} close: {moves}."
            )
            if sector in day5_rank:
                text += (f" At Day +5 this sector ranked {day5_rank.index(sector) + 1} "
                         f"of {len(day5_rank)} by return (1 = lowest return).")
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
    return docs + summary_documents(docs) + analysis_documents(data.get("analysis"))


def comove_text(c) -> str:
    if not c:
        return ""
    verdict = ("clearly more than on normal days" if c["p_value"] < 0.05 else
               "a little more than on normal days, but not clearly beyond chance" if c["event_mean_pct"] > c["normal_mean_pct"]
               else "no more than on normal days")
    return (f"From the trading day before the news peak to the day after, the VIX changed by "
            f"{c['event_mean_pct']:+.2f}% on average around the {c['events']} spikes (median {c['event_median_pct']:+.2f}%), "
            f"versus {c['normal_mean_pct']:+.2f}% for the same 2-day window on normal days, so it rose {verdict} "
            f"(p = {c['p_value']}, random-date test). It rose in {c['events_vix_up']} of {c['events']} spikes, "
            f"versus {c['normal_share_up']:.0%} of normal windows. ")


def analysis_documents(a) -> list:
    """Cross-event analyses from data_collection/build_analysis.py, written
    answer-first so the embedding model (first ~256 tokens) sees the answer."""
    if not a:
        return []
    docs = []
    mk = lambda key, label, text: docs.append({
        "doc_id": f"analysis:{key}", "type": "analysis", "event_date": "all events", "label": label,
        "sector": None, "source_url": "", "urls": [], "text": text})

    s = a["stocks"]
    fmt = lambda rows: "; ".join(f"{r['ticker']} ({r['sector'] or 'S&P 500'}) {fmt_pct(r['avg_return_pct'])}, "
                                 f"rose in {r['rose_in']}" for r in rows)
    mk("stocks", "Best and worst stocks after spikes",
       "Which stocks did best and worst after geopolitical tension spikes, on average across all detected "
       f"events (historical, not a recommendation). Compared {s['day5']['stocks_compared']} S&P 500 stocks. "
       f"Best average return at Day +5: {fmt(s['day5']['best'][:6])}. "
       f"Worst average return at Day +5: {fmt(s['day5']['worst'][:6])}. "
       f"Best average return at Day +30: {fmt(s['day30']['best'][:5])}. "
       f"Worst average return at Day +30: {fmt(s['day30']['worst'][:5])}.")

    v = a["volatile"]
    similar = ("somewhat similar" if v["mean_jaccard"] >= 2 * v["random_jaccard"] else "mostly different")
    mk("volatile", "Were the biggest movers similar across events",
       f"Were the most volatile stocks similar across geopolitical events? The {v['movers_per_event']} stocks "
       f"with the largest Day +5 moves (up or down) were {similar} from event to event: two events shared "
       f"{v['mean_jaccard']:.1%} of their biggest movers on average (Jaccard overlap), versus "
       f"{v['random_jaccard']:.1%} for two random sets of {v['movers_per_event']} out of {v['stocks']} stocks. "
       "Stocks that were among the biggest movers most often: "
       + "; ".join(f"{r['ticker']} in {r['events']} of {v['events']} events" for r in v["most_repeated"][:10]) + ".")

    if "sectors" in a:
        sx = a["sectors"]
        parts = []
        for g, res in sx["tests"].items():
            sig = [f"{sec} {r['event_mean_pct']:+.2f}% (p = {r['p_value']})" for sec, r in res.items() if r["p_value"] < 0.05]
            allp = "; ".join(f"{sec} {r['event_mean_pct']:+.2f}% vs {r['normal_mean_pct']:+.2f}% normally (p = {r['p_value']})"
                             for sec, r in res.items())
            parts.append(f"After {g} ({sx['groups'][g]} events): "
                         + (f"clearly different from normal days: {', '.join(sig)}. " if sig else
                            "no sector moved clearly differently from normal days. ")
                         + f"All sectors: {allp}.")
        mk("sectors", "How sectors moved after spikes, by event type",
           "How did sectors move after geopolitical spikes, compared with normal days and by type of event? "
           f"Excess return = sector minus the average S&P 500 stock, from the day before the news peak's first "
           f"trading day to {sx['hold_days']} trading days later; p-values from a random-date test (below 0.05 = "
           "unlikely to be chance). " + " ".join(parts))

    if "vix_lead" in a:
        x = a["vix_lead"]
        corr = x["corr_tension_vs_vix_change_k_days_later"]
        best_k = max(corr, key=lambda k: abs(corr[k]))
        who = ("the news usually came first and the VIX jumped afterwards" if x["median_days"] > 0 else
               "the VIX usually jumped before the news peaked" if x["median_days"] < 0 else
               "the VIX usually jumped on the first trading day after the news peak")
        mk("vix_lead", "Did the VIX or the Geo-Tension score move first",
           f"Did the VIX and the Geo-Tension Index move together? {comove_text(x.get('comove'))}"
           f"Did one move first? Around the {x['events']} detected spikes, {who}. "
           f"The VIX's biggest daily jump within {x['window_trading_days']} trading days of the news peak came a "
           f"median of {x['median_days']:+.1f} trading days after it (day 0 = the first trading day on or after the "
           f"news peak, so weekend news isn't counted as a delay). The VIX jumped before the news peak in "
           f"{x['vix_before']} events, on day 0 in {x['vix_same_day']}, and later in {x['vix_after']}. "
           f"Over all days, the tension score's correlation with the VIX's daily change k days later was "
           f"strongest at k = {best_k} ({corr[best_k]:+.3f}); correlations for k = -2..+2: "
           + ", ".join(f"k={k}: {corr[str(k)] if str(k) in corr else corr[k]:+.3f}" for k in range(-2, 3))
           + ". Correlation, not causation.")
    return docs


def summary_documents(docs: list) -> list:
    """Comparison documents for questions that span ALL events ("after which
    spike did Airlines fall the most?"). Top-k retrieval can't compare 27
    event documents, so the comparison is precomputed here, from the same
    numbers, with the answer sentences first (the embedding model reads only
    the first ~256 tokens)."""
    out = []
    events = [d for d in docs if d["type"] == "event"]
    label = {d["event_date"]: d["label"] for d in events}
    sector_docs = [d for d in docs if d["type"] == "event_sector"]

    for sector in sorted({d["sector"] for d in sector_docs}):
        rows = [(d["event_date"], d["returns"]) for d in sector_docs if d["sector"] == sector]
        parts = [f"Comparison across all {len(rows)} spike events: {sector} stocks."]
        for h in (5, 30, 1):
            vals = sorted((r[str(h)]["return_pct"], date) for date, r in rows if str(h) in r)
            if not vals:
                continue
            (lo, lo_d), (hi, hi_d) = vals[0], vals[-1]
            parts.append(f"At Day +{h}, {sector} fell the most after the spike on {lo_d} "
                         f"({label[lo_d]}, {fmt_pct(lo)}) and rose the most after the spike on "
                         f"{hi_d} ({label[hi_d]}, {fmt_pct(hi)}).")
        out.append({"doc_id": f"summary:{sector}", "type": "sector_summary", "event_date": "all events",
                    "label": f"{sector} across all spikes", "sector": sector, "source_url": "",
                    "urls": [], "text": " ".join(parts)})

    if events and all(e.get("z_score") is not None for e in events):
        parts = [f"Overview of all {len(events)} detected spike events."]
        top = max(events, key=lambda e: e["z_score"])
        parts.append(f"Largest z-score overall: {top['event_date']} ({top['label']}, z-score {top['z_score']:.2f}).")
        by_year = {}
        for e in events:
            y = e["event_date"][:4]
            if y not in by_year or e["z_score"] > by_year[y]["z_score"]:
                by_year[y] = e
        parts.append("Largest z-score in each year: " + "; ".join(
            f"{y}: {e['event_date']} ({e['label']}, {e['z_score']:.2f})" for y, e in sorted(by_year.items())) + ".")
        out.append({"doc_id": "summary:events", "type": "events_summary", "event_date": "all events",
                    "label": "All detected spikes", "sector": None, "source_url": "", "urls": [],
                    "text": " ".join(parts)})
    return out


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
    n = lambda t: sum(d["type"] == t for d in docs)
    print(f"Wrote {len(docs)} documents ({n('event')} events, {n('event_sector')} event-sector, "
          f"{n('sector_summary') + n('events_summary')} comparison summaries) to {args.out}")


if __name__ == "__main__":
    main()
