"""Helpers shared by the data pipeline scripts."""
import re

import pandas as pd

PRICE_WINDOW = 30  # calendar days before/after a spike (same as the original event_window.py)


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


def build_reactions(closes: pd.DataFrame, dates) -> pd.DataFrame:
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
    for s in dates:
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
            "ticker": win["ticker"],
            "days_from_spike": (win["date"] - sd).dt.days.astype(int),
            "avg_return_pct": win["return_pct"],
            "num_obs": 1,
        }))
    return pd.concat(out, ignore_index=True)
