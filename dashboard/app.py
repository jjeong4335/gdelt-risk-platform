import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import requests
import zipfile
import io
from datetime import datetime
import yfinance as yf
import sys
from pathlib import Path
# Make the repo root importable so the dashboard can use the rag/ package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard.constants import SECTOR_MAP, SPIKE_LABELS

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Geopolitical Risk Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.markdown('<meta http-equiv="refresh" content="900">', unsafe_allow_html=True)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 16px;
    }
    .metric-label { font-size: 12px; color: #888; margin-bottom: 4px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 32px; font-weight: 700; color: #1a1a1a; line-height: 1.1; }
    .metric-delta-pos { font-size: 13px; color: #16a34a; }
    .metric-delta-neg { font-size: 13px; color: #dc2626; }
    .metric-delta-neu { font-size: 13px; color: #888; }
    .risk-high { color: #dc2626; font-size: 28px; font-weight: 700; }
    .risk-medium { color: #f59e0b; font-size: 28px; font-weight: 700; }
    .risk-low { color: #16a34a; font-size: 28px; font-weight: 700; }
    .live-badge { background: #dcfce7; color: #16a34a; border-radius: 20px; padding: 4px 12px; font-size: 13px; font-weight: 600; display: inline-block; }
    .news-time { font-size: 11px; color: #888; margin-bottom: 2px; }
    .news-item { padding: 8px 0; border-bottom: 1px solid #f0f0f0; }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-right: 6px; }
    .tag-military { background: #fee2e2; color: #dc2626; }
    .tag-sanctions { background: #fef3c7; color: #d97706; }
    .tag-diplomatic { background: #dbeafe; color: #2563eb; }
    .tag-nuclear { background: #f3e8ff; color: #7c3aed; }
    .tag-political { background: #ffedd5; color: #ea580c; }
    .tag-other { background: #f1f5f9; color: #64748b; }
    .sector-row { display: flex; align-items: center; margin-bottom: 10px; }
    .sector-name { width: 120px; font-size: 13px; color: #444; }
    .sector-bar-pos { background: #86efac; height: 8px; border-radius: 4px; }
    .sector-bar-neg { background: #fca5a5; height: 8px; border-radius: 4px; }
    .sector-val-pos { color: #16a34a; font-size: 13px; font-weight: 600; margin-left: 8px; }
    .sector-val-neg { color: #dc2626; font-size: 13px; font-weight: 600; margin-left: 8px; }
    .port-row { display: flex; align-items: center; padding: 6px 0; font-size: 13px; }
    .port-ticker { font-weight: 700; width: 50px; }
    .port-sector { color: #888; width: 90px; }
    .port-weight { color: #444; width: 40px; }
    .port-return-pos { color: #16a34a; font-weight: 600; margin-left: auto; }
    .port-return-neg { color: #dc2626; font-weight: 600; margin-left: auto; }
    .summary-row { display: flex; justify-content: space-between; padding: 6px 0; font-size: 13px; border-top: 1px solid #f0f0f0; }
    .summary-label { color: #666; }
    .summary-val-pos { color: #16a34a; font-weight: 600; }
    .summary-val-neg { color: #dc2626; font-weight: 600; }
    .chart-title { font-size: 13px; color: #444; font-weight: 500; margin-bottom: 8px; }
    .block-container { padding-top: 2rem; }
    /* bordered st.container() blocks look like the HTML cards */
    .st-key-live_chart_card { background: white; border-radius: 12px; border: none !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); padding: 20px 24px !important; margin-bottom: 16px; }
    .muted { color:#888; font-size:13px; }
    .sector-legend { font-size:11px; color:#888; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# Data folder: GEO_DATA_DIR if set; else the original cluster folder if it
# exists; else the laptop rebuild made by data_collection/rebuild_for_rag.py.
_CLUSTER_DIR = "/home/jj4335_nyu_edu/dashboard_data"
DATA_DIR = os.path.expanduser(os.environ.get("GEO_DATA_DIR") or (
    _CLUSTER_DIR if os.path.isdir(_CLUSTER_DIR) else "~/dashboard_data_rebuilt"))
# The rebuild has no 10-year daily index (it needs the full 2 TB archive), so
# the parts of the dashboard that need it say so instead of crashing.
HAS_HISTORY = os.path.exists(f"{DATA_DIR}/geo_tension_index.parquet")

# ── Keyword filter for supporting news ───────────────────────────
import re as _re

_STOPWORDS = {"war", "tensions", "protests", "crisis", "escalation", "airstrikes", "peak", "ceasefire", "civil", "political", "geopolitical", "border", "strait"}
_CONFLICT_CONTEXT = {"attack", "missile", "strike", "military", "troops", "invasion", "bomb", "nuclear", "riot", "clash", "conflict", "threat", "shoot", "kill", "dead", "wound", "protest", "warn", "launch"}

def _extract_primary_keywords(label):
    """Extract proper nouns from spike label, excluding generic conflict words."""
    words = _re.findall(r'[A-Z][a-z]+', label)
    return [w.lower() for w in words if w.lower() not in _STOPWORDS]

def _passes_news_filter(title, label):
    """Return True if title contains a primary keyword AND a conflict context keyword."""
    if not isinstance(title, str):
        return False
    primary = _extract_primary_keywords(label)
    t = title.lower()
    if not primary:
        return True
    return any(k in t for k in primary) and any(k in t for k in _CONFLICT_CONTEXT)

# ── Load data ─────────────────────────────────────────────────────
@st.cache_data
def load_geo_tension():
    if not HAS_HISTORY:
        return pd.DataFrame({"date": pd.to_datetime([]), "geo_tension_index": [], "total_events": []})
    df = pd.read_parquet(f"{DATA_DIR}/geo_tension_index.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year.between(2016, 2026)]
    return df.sort_values("date")

@st.cache_data
def load_spike_events():
    df = pd.read_parquet(f"{DATA_DIR}/spike_events.parquet")
    df["date"] = pd.to_datetime(df["date"])
    if "geo_tension_index" not in df:  # quick-sample rebuild has no index values
        df["geo_tension_index"] = float("nan")
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    # Auto-label from spike_news keywords if not in SPIKE_LABELS
    try:
        spike_news_df = pd.read_parquet(f"{DATA_DIR}/spike_news/")
        spike_news_df["article_date"] = pd.to_datetime(spike_news_df["article_date"]).dt.date
        import collections, re
        STOPWORDS = {"the","and","for","are","was","has","had","with","that","this",
                     "from","have","been","will","its","not","but","they","their",
                     "said","says","over","after","more","also","than","into","about"}
        def auto_label(date_str):
            if date_str in SPIKE_LABELS:
                return SPIKE_LABELS[date_str]
            import datetime
            dt = datetime.date.fromisoformat(date_str)
            news = spike_news_df[spike_news_df["article_date"].apply(lambda d: abs((d-dt).days)<=3)]
            titles = " ".join(news["title"].dropna().tolist()).lower()
            words = re.findall(r"[a-z]{4,}", titles)
            freq = collections.Counter(w for w in words if w not in STOPWORDS)
            top = [w.capitalize() for w, _ in freq.most_common(3)]
            return " · ".join(top) if top else "Geopolitical Event"
        df["label"] = df["date_str"].apply(auto_label)
    except:
        df["label"] = df["date_str"].map(SPIKE_LABELS).fillna("Geopolitical Event")
    return df.sort_values("geo_tension_index", ascending=False)

@st.cache_data
def load_ticker_summary():
    if os.path.exists(f"{DATA_DIR}/ticker_summary.parquet"):
        return pd.read_parquet(f"{DATA_DIR}/ticker_summary.parquet")
    # Rebuilt data: same aggregation as event_window.py, done in pandas.
    r = load_ticker_reaction_by_spike()
    g = r.groupby("ticker")
    return pd.DataFrame({
        "avg_return_5d": r[r["days_from_spike"] == 5].groupby("ticker")["avg_return_pct"].mean(),
        "avg_return_30d": r[r["days_from_spike"] == 30].groupby("ticker")["avg_return_pct"].mean(),
        "worst_drawdown": g["avg_return_pct"].min(),
        "num_events": g["spike_date"].nunique(),
    }).reset_index()


@st.cache_data
def load_ticker_reaction_by_spike():
    """Load spike-level ticker reaction data."""
    df = pd.read_parquet(f"{DATA_DIR}/ticker_reaction_by_spike/")
    df["spike_date"] = pd.to_datetime(df["spike_date"]).dt.date
    return df
@st.cache_data(ttl=900)
def fetch_vix():
    try:
        vix = yf.download("^VIX", start="2016-01-01", auto_adjust=True, progress=False)
        vix = vix[["Close"]].reset_index()
        vix.columns = ["date", "vix"]
        vix["date"] = pd.to_datetime(vix["date"])
        return vix
    except:
        return pd.DataFrame(columns=["date", "vix"])

# ── Fetch live ETF sector returns (1-min interval) ───────────────
SECTOR_ETF_MAP = {
    "Airlines": "JETS",
    "Semiconductors": "SOXX",
    "Energy": "XLE",
    "Defense": "ITA",
    "Tech": "XLK",
    "Gold/Safe Haven": "GLD",
}

@st.cache_data(ttl=60)
def fetch_live_sector_returns():
    """Fetch todays intraday return for each sector ETF via yfinance."""
    rows = []
    for sector, ticker in SECTOR_ETF_MAP.items():
        try:
            df = yf.download(ticker, period="1d", interval="1m", progress=False)
            if len(df) >= 2:
                open_price = df["Close"].iloc[0]
                latest_price = df["Close"].iloc[-1]
                ret = float((latest_price - open_price) / open_price * 100)
            else:
                ret = None
        except Exception:
            ret = None
        rows.append({"sector": sector, "avg": ret})
    return pd.DataFrame(rows).sort_values("avg", ascending=False)

@st.cache_data(ttl=900)
def fetch_latest_gdelt_news():
    try:
        from dashboard.collect_live import latest_gkg_zip
        content, _ = latest_gkg_zip()  # steps back 15 min if the newest file isn't up yet
        z = zipfile.ZipFile(io.BytesIO(content))
        fname = z.namelist()[0]
        with z.open(fname) as f:
            df = pd.read_csv(f, sep="\t", header=None, on_bad_lines="skip",
                           usecols=[1, 3, 4, 7],
                           names=["date", "source", "url", "themes"])

        # Filter geopolitical keywords - more specific themes
        keywords = "MILITARY|WAR|WEAPON|MISSILE|SANCTION|EMBARGO|COUP|PROTEST|DIPLOMATIC|NUCLEAR|CYBERATTACK"
        df = df[df["themes"].str.contains(keywords, na=False, case=False)]
        # Additional title-level filter: must contain geopolitical keywords
        title_keywords = "war|attack|sanction|missile|nuclear|military|conflict|invasion|protest|coup|strike|troops|weapon|crisis|threat|bomb|terror|drone|ceasefire|diplomacy|nato|un |iran|russia|china|israel|ukraine|gaza|korea"
        df = df[df["url"].str.lower().str.contains(title_keywords, na=False)]

        def get_category(themes):
            t = str(themes)
            if any(k in t for k in ["MILITARY", "WAR", "WEAPON", "MISSILE", "INVASION"]):
                return "Military"
            elif any(k in t for k in ["SANCTION", "EMBARGO", "TARIFF"]):
                return "Sanctions"
            elif any(k in t for k in ["DIPLOMATIC", "EXPULSION"]):
                return "Diplomatic"
            elif "NUCLEAR" in t:
                return "Nuclear"
            elif any(k in t for k in ["COUP", "PROTEST"]):
                return "Political"
            return "Other"

        def extract_title(url):
            try:
                parts = str(url).rstrip("/").split("/")
                slug = parts[-1].split(".")[0]
                title = slug.replace("-", " ").replace("_", " ").title()
                return title[:150] if title else str(url)[:150]
            except:
                return str(url)[:60]

        df["category"] = df["themes"].apply(get_category)
        df["title"] = df["url"].apply(extract_title)
        # Filter out junk titles: too short, numeric-only, hex strings, no spaces
        df = df[
            df["title"].str.len() > 20
        ]
        df = df[df["title"].str.contains(" ", na=False)]  # must have spaces
        df = df[~df["title"].str.match(r"^[A-Fa-f0-9\s]+$", na=False)]  # no hex
        df = df[~df["title"].str.match(r"^[A-Z][a-z]* \d+$", na=False)]  # no "Article 893559"
        df = df[df["title"].str.replace(r"[^a-zA-Z]", "", regex=True).str.len() > 10]  # enough letters
        df["time"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d%H%M%S", errors="coerce", utc=True).dt.tz_convert("America/New_York")
        df = df.dropna(subset=["time"])
        df["time"] = df["time"].dt.strftime("%H:%M ET")
        df = df.sort_values("time", ascending=False)
        return df.head(10)[["time", "title", "source", "category"]].reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame(columns=["time", "title", "source", "category"])

@st.cache_data(ttl=900)
def fetch_live_tension():
    """Score the latest 15-minute GDELT file the same way collect_live.py does.
    Used for the headline card when there is no daily history to show."""
    try:
        from dashboard.collect_live import compute_tension, fetch_latest_gdelt
        df = fetch_latest_gdelt()
        if df is None:
            return None, 0
        intl = df[df[6].astype(str).str.contains(r"#(?!US)[A-Z]{2}#", regex=True, na=False)]
        return compute_tension(df), len(intl)
    except Exception:
        return None, 0


# Load all data
geo_tension = load_geo_tension()
spike_events = load_spike_events()
ticker_summary = load_ticker_summary()
vix_df = fetch_vix()
news_df = fetch_latest_gdelt_news()

# ── Compute metrics ───────────────────────────────────────────────
if HAS_HISTORY:
    latest = geo_tension.iloc[-1]
    prev = geo_tension.iloc[-2]
    current_score = latest["geo_tension_index"]
    delta_score = current_score - prev["geo_tension_index"]
    mean = geo_tension["geo_tension_index"].mean()
    std = geo_tension["geo_tension_index"].std()
    threshold = mean + 3 * std

    if current_score > threshold:
        risk_level, risk_class = "High", "risk-high"
    elif current_score > mean + 1.5 * std:
        risk_level, risk_class = "Medium", "risk-medium"
    else:
        risk_level, risk_class = "Low", "risk-low"

    events_today = int(latest["total_events"]) if "total_events" in latest else 0
    score_note = f"{'+' if delta_score >= 0 else ''}{delta_score:.1f} since yesterday"
    threshold_note = f"Threshold: {threshold:.1f}"
    events_note = "Geopolitical news volume"
else:
    live_score, live_events = fetch_live_tension()
    current_score = live_score if live_score is not None else float("nan")
    delta_score = 0.0
    risk_level, risk_class = "n/a", "metric-value"
    events_today = live_events
    score_note = ("Latest 15-min GDELT file" if live_score is not None
                  else "Live GDELT feed unavailable right now")
    threshold_note = "Needs the 10-year daily history"
    events_note = "Intl. articles, latest 15 min"

sector_df = fetch_live_sector_returns()







default_portfolio = [("AAPL", "Tech", 30), ("XOM", "Energy", 50), ("LMT", "Defense", 20)]
port_stats = []
for ticker, sector, weight in default_portfolio:
    row = ticker_summary[ticker_summary["ticker"] == ticker]
    r5 = row.iloc[0]["avg_return_5d"] if len(row) > 0 else 0
    wd = row.iloc[0]["worst_drawdown"] if len(row) > 0 else 0
    port_stats.append({"ticker": ticker, "sector": sector, "weight": weight, "r5": r5, "wd": wd})

port_df = pd.DataFrame(port_stats)
weighted_5d = (port_df["r5"] * port_df["weight"] / 100).sum()
weighted_wd = (port_df["wd"] * port_df["weight"] / 100).sum()
most_exposed = port_df.loc[(port_df["wd"].abs() * port_df["weight"] / 100).idxmax(), "ticker"]

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 Live Dashboard", "📅 Historical Events", "💬 Q&A"])

# ══════════════════════════════════════════════════════════════════
# TAB 1: LIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab1:
    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.markdown("## 🌍 Geopolitical Risk Dashboard")
        st.caption("VIX tells you how scared the market is. This tells you why — and how exposed your portfolio is.")
    with col_badge:
        st.markdown("""
        <div style='text-align:right; padding-top:12px'>
        </div>
        """, unsafe_allow_html=True)

    # Top 4 metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        delta_class = ("metric-delta-neu" if not HAS_HISTORY
                       else "metric-delta-pos" if delta_score >= 0 else "metric-delta-neg")
        sign = "+" if delta_score >= 0 else ""
        st.markdown(f"""<div class='card'>
            <div class='metric-label'>Geo-Tension Index</div>
            <div class='metric-value'>{"—" if current_score != current_score else f"{current_score:.1f}"}</div>
            <div class='{delta_class}'>{score_note}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class='card'>
            <div class='metric-label'>Risk Level</div>
            <div class='{risk_class}'>{risk_level}</div>
            <div class='metric-delta-neu'>{threshold_note}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class='card'>
            <div class='metric-label'>Events Today</div>
            <div class='metric-value'>{events_today:,}</div>
            <div class='metric-delta-neu'>{events_note}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        exp_class = "metric-delta-pos" if weighted_5d >= 0 else "metric-delta-neg"
        exp_sign = "+" if weighted_5d >= 0 else ""
        st.markdown(f"""<div class='card'>
            <div class='metric-label'>Portfolio Exposure</div>
            <div class='metric-value'>{exp_sign}{weighted_5d:.1f}%</div>
            <div class='{exp_class}'>Hist. avg on similar events</div>
        </div>""", unsafe_allow_html=True)

    # Chart + News
    col_chart, col_news = st.columns([1, 1])

    with col_chart:
        fig = go.Figure()
        # Check if US market is open (Mon-Fri 9:30-16:00 ET)
        from datetime import datetime
        import pytz
        et = pytz.timezone("America/New_York")
        now_et = datetime.now(et)
        market_open = (
            now_et.weekday() < 5 and
            (now_et.hour, now_et.minute) >= (9, 30) and
            now_et.hour < 16
        )
        if True:  # Always show live tension
            # Live mode: show today's GDELT tension
            import sqlite3
            DB_PATH = f"{DATA_DIR}/live_tension.db"
            try:
                conn = sqlite3.connect(DB_PATH)
                live_df = pd.read_sql("SELECT timestamp, tension_score FROM live_tension WHERE date(timestamp) >= date('now') ORDER BY timestamp", conn, parse_dates=["timestamp"])
                conn.close()
            except:
                live_df = pd.DataFrame(columns=["timestamp", "tension_score"])







                live_df = pd.DataFrame(columns=["timestamp", "tension_score"])












            if len(live_df) == 0 and not HAS_HISTORY and current_score == current_score:
                # No live collector running: show the one score computed above.
                live_df = pd.DataFrame({"timestamp": [now_et.replace(tzinfo=None)],
                                        "tension_score": [current_score]})
            if len(live_df) > 0:
                fig.add_trace(go.Scatter(
                    x=live_df["timestamp"], y=live_df["tension_score"],
                    mode="lines+markers", fill="tozeroy", name="Geo-Tension Index",
                    line=dict(color="#3b82f6", width=2),
                    fillcolor="rgba(59,130,246,0.1)",
                    yaxis="y1"
                ))
            # VIX intraday
            try:
                import yfinance as yf
                vix_live = yf.download("^VIX", period="1d", interval="5m", progress=False)
                if len(vix_live) > 0:
                    vix_live = vix_live["Close"].reset_index()
                    vix_live.columns = ["datetime", "vix"]
                    vix_live["datetime"] = pd.to_datetime(vix_live["datetime"], utc=True).dt.tz_localize(None)
                    vix_live = vix_live[(vix_live["datetime"].dt.hour >= 9) & ((vix_live["datetime"].dt.hour > 9) | (vix_live["datetime"].dt.minute >= 30)) & (vix_live["datetime"].dt.hour < 16)]
                    fig.add_trace(go.Scatter(
                        x=vix_live["datetime"], y=vix_live["vix"],
                        mode="lines", name="VIX",
                        line=dict(color="#f59e0b", width=2, dash="dot"),
                        yaxis="y2"
                    ))
            except:
                pass
            n_points = len(live_df)
            chart_title = f"Geo-Tension Index — Today ({n_points} readings, updates every 15 min)"
            fig.update_layout(
                height=250, margin=dict(l=0, r=40, t=10, b=0),
                xaxis=dict(showgrid=False, tickformat="%H:%M"),
                yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Tension", side="left"),
                yaxis2=dict(overlaying="y", side="right", title="VIX", showgrid=False),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=1.1, x=0)
            )
        else:
            # Market closed: show historical chart
            cutoff_5d = pd.Timestamp.now() - pd.Timedelta(days=5)
            geo_5d = geo_tension[geo_tension["date"] >= cutoff_5d]
            vix_5d = vix_df[vix_df["date"] >= cutoff_5d] if len(vix_df) > 0 else vix_df
            fig.add_trace(go.Scatter(
                x=geo_5d["date"], y=geo_5d["geo_tension_index"],
                mode="lines+markers", fill="tozeroy", name="Geo-Tension Index",
                line=dict(color="#3b82f6", width=2),
                fillcolor="rgba(59,130,246,0.1)",
                yaxis="y1"
            ))
            if len(vix_5d) > 0:
                fig.add_trace(go.Scatter(
                    x=vix_5d["date"], y=vix_5d["vix"],
                    mode="lines+markers", name="VIX",
                    line=dict(color="#f59e0b", width=2, dash="dot"),
                    yaxis="y2"
                ))
            chart_title = "Geo-Tension Index vs VIX — Last 5 Days (Market Closed)"
            fig.update_layout(
                height=250, margin=dict(l=0, r=40, t=10, b=0),
                xaxis=dict(showgrid=False, tickformat="%m/%d"),
                yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Tension", side="left"),
                yaxis2=dict(overlaying="y", side="right", title="VIX", showgrid=False),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=1.1, x=0)
            )
        with st.container(border=True, key="live_chart_card"):
            st.markdown(f"<div class='chart-title'>{chart_title}</div>", unsafe_allow_html=True)
            if len(fig.data) == 0:
                st.markdown("<p class='muted'>No live readings yet. The tension score and VIX "
                            "appear here once GDELT and market data load.</p>", unsafe_allow_html=True)
            else:
                st.plotly_chart(fig, width="stretch")
        # Sector + Portfolio



        live_ok = sector_df["avg"].notna().any()
        sector_df = sector_df.dropna(subset=["avg"])
        max_abs = sector_df["avg"].abs().max() if live_ok else 0
        sector_title = "Sector reaction — live today · refreshes every 1 min" if market_open else "Sector reaction — last trading day"
        sector_html = f"<div class='card'><div class='chart-title'>{sector_title}</div>"
        for _, row in sector_df.iterrows():
            v = row["avg"]
            bar_w = int(abs(v) / max_abs * 180) if max_abs > 0 else 0
            bar_cls = "sector-bar-pos" if v >= 0 else "sector-bar-neg"
            val_cls = "sector-val-pos" if v >= 0 else "sector-val-neg"
            sign = "+" if v >= 0 else ""
            sector_html += f"""<div class='sector-row'>
                <div class='sector-name'>{row['sector']}</div>
                <div class='{bar_cls}' style='width:{bar_w}px'></div>
                <div class='{val_cls}'>{sign}{v:.1f}%</div>
            </div>"""
        if not live_ok:
            sector_html += "<p class='muted'>Live ETF prices are unavailable right now.</p>"
        else:
            sector_html += ("<div class='sector-legend'>Sector ETFs (JETS, SOXX, XLE, ITA, XLK, GLD), "
                            "change since the open · green = up, red = down</div>")
        sector_html += "</div>"
        st.markdown(sector_html, unsafe_allow_html=True)


    with col_news:
        tag_classes = {
            "Military": "tag-military", "Sanctions": "tag-sanctions",
            "Diplomatic": "tag-diplomatic", "Nuclear": "tag-nuclear",
            "Political": "tag-political", "Other": "tag-other"
        }
        news_html = "<div class='card' style='height:100%'><div class='chart-title'>Latest news events · updates every 15 min</div>"
        # Filter out low-quality titles: hex UUIDs, URLs, short or symbol-heavy strings
        news_df_filtered = news_df[
            (news_df["title"].str.replace(r"[\d\s]", "", regex=True).str.len() > 5) &
            (~news_df["title"].str.contains(r"#|\.com|\.org|\.net", na=False)) &
            (~news_df["title"].str.contains(r"[A-Fa-f0-9]{6,}", regex=True, na=False)) &
            (news_df["title"].str.contains(" ", na=False)) &
            (news_df["title"].str.replace(r"[^a-zA-Z]", "", regex=True).str.len() > 15)
        ] if len(news_df) > 0 else news_df
        if len(news_df_filtered) > 0:
            for _, row in news_df_filtered.head(6).iterrows():
                time_str = row["time"] if pd.notna(row["time"]) else ""
                cat = row["category"]
                tag_cls = tag_classes.get(cat, "tag-other")
                title = str(row["title"]) if pd.notna(row["title"]) else "Unknown"
                source = str(row["source"]) if pd.notna(row["source"]) else ""
                news_html += f"""<div class='news-item'>
                    <div class='news-time'>{time_str} · {source}</div>
                    <span class='tag {tag_cls}'>{cat}</span>{title}
                </div>"""
        elif len(news_df) == 0:
            news_html += "<p style='color:#888;font-size:13px'>No recent events found.</p>"
        news_html += "</div>"
        st.markdown(news_html, unsafe_allow_html=True)
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📅 Historical Geopolitical Spike Events")
    st.caption("31 major geopolitical events detected from GDELT news sentiment (2016–2026)")

    if not HAS_HISTORY:
        st.info("The 10-year daily index isn't available in the rebuilt data (it needs the full "
                "GDELT archive). The chart shows the 31 spike days, with each day's index "
                "recomputed from every GDELT file for that day.")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=geo_tension["date"], y=geo_tension["geo_tension_index"],
        mode="lines", name="Geo-Tension Index",
        line=dict(color="#3b82f6", width=1.5),
        fillcolor="rgba(59,130,246,0.1)", fill="tozeroy"
    ))
    fig_hist.add_trace(go.Scatter(
        x=spike_events["date"], y=spike_events["geo_tension_index"],
        mode="markers", name="Spike Events",
        marker=dict(color="#dc2626", size=8),
        text=spike_events["label"],
        hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Score: %{y:.2f}<extra></extra>"
    ))
    # Show 5 years by default with rangeslider to scroll through all data
    end_date   = geo_tension["date"].max() if HAS_HISTORY else spike_events["date"].max() + pd.Timedelta(days=30)
    start_date = end_date - pd.Timedelta(days=365*5) if HAS_HISTORY else pd.Timestamp("2017-10-01")

    fig_hist.update_layout(
        height=400,
        xaxis=dict(
            showgrid=False, title="Date",
            range=[start_date, end_date],
            minallowed="2016-01-01",
            maxallowed=str(end_date.date()),
        ),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Tension Score", rangemode="nonnegative", fixedrange=True),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.05),
        hovermode="closest"
    )
    st.plotly_chart(fig_hist, width="stretch")
    st.divider()
    st.subheader("🔍 Explore Spike Event")
    spike_options = (spike_events.sort_values("date", ascending=False)
                     .apply(lambda r: f"{r['date'].strftime('%Y-%m-%d')} | {r['label']}", axis=1)
                     .tolist())
    selected_option = st.selectbox("Select a spike event to analyze", spike_options)
    selected_date  = selected_option.split(" | ")[0]
    selected_event = selected_option.split(" | ")[1]
    import datetime
    if selected_date:
        st.divider()
        st.markdown(f"### 📌 {selected_event} ({selected_date})")

        col_left, col_right = st.columns([1, 1])

        # ── Left: ±5 day Geo-Tension + VIX chart ─────────────────────────
        with col_left:
            st.markdown("#### 📈 Geo-Tension & VIX (±5 days)")
            if not HAS_HISTORY:
                st.caption("Daily index around the spike isn't in the rebuilt data; showing VIX only.")
            spike_dt = pd.Timestamp(selected_date)
            start_dt = spike_dt - pd.Timedelta(days=5)
            end_dt   = spike_dt + pd.Timedelta(days=5)
            geo_window = geo_tension[
                (geo_tension["date"] >= start_dt) & (geo_tension["date"] <= end_dt)
            ]
            vix_window = vix_df[
                (vix_df["date"] >= start_dt) & (vix_df["date"] <= end_dt)
            ] if len(vix_df) > 0 else pd.DataFrame()

            fig_event = go.Figure()
            fig_event.add_trace(go.Scatter(
                x=geo_window["date"], y=geo_window["geo_tension_index"],
                mode="lines+markers", name="Geo-Tension",
                line=dict(color="#3b82f6", width=2),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
                yaxis="y1"
            ))
            if len(vix_window) > 0:
                fig_event.add_trace(go.Scatter(
                    x=vix_window["date"], y=vix_window["vix"],
                    mode="lines+markers", name="VIX",
                    line=dict(color="#f59e0b", width=2, dash="dot"),
                    yaxis="y2"
                ))

            fig_event.update_layout(
                height=280,
                margin=dict(l=0, r=40, t=10, b=0),
                xaxis=dict(showgrid=False, tickformat="%m/%d"),
                yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Tension"),
                yaxis2=dict(overlaying="y", side="right", title="VIX", showgrid=False),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=1.1, x=0)
            )
            st.plotly_chart(fig_event, width="stretch")

            # Sector Impact
            st.markdown("#### 📊 Sector Impact (Day +5)")
            try:
                reaction_df  = load_ticker_reaction_by_spike()
                spike_date_d = datetime.date.fromisoformat(selected_date)
                spike_data   = reaction_df[reaction_df["spike_date"] == spike_date_d].copy()
                # If no data for exact date, use nearest available date
                if len(spike_data) == 0:
                    available = reaction_df["spike_date"].unique()
                    nearest = min(available, key=lambda d: abs((d - spike_date_d).days))
                    spike_data = reaction_df[reaction_df["spike_date"] == nearest].copy()
                    st.caption(f"ℹ️ No trading data for {selected_date}. Showing nearest date: {nearest}")
                ticker_to_sector = {t: s for s, tickers in SECTOR_MAP.items() for t in tickers}
                spike_data["sector"] = spike_data["ticker"].map(ticker_to_sector)
                spike_data = spike_data.dropna(subset=["sector"])
                # Use day 5 or nearest available trading day
                available_days = sorted(spike_data[spike_data["days_from_spike"] > 0]["days_from_spike"].unique())
                target_day = min(available_days, key=lambda x: abs(x-5)) if available_days else None
                if target_day:
                    sector_5d = spike_data[spike_data["days_from_spike"] == target_day].groupby("sector")["avg_return_pct"].mean().sort_values()
                else:
                    sector_5d = pd.Series(dtype=float)
                if not sector_5d.empty:
                    max_abs = sector_5d.abs().max()
                    sec_html = "<div class='card'>"
                    for sector, ret in sector_5d.items():
                        bar_w = int(abs(ret) / max_abs * 180) if max_abs > 0 else 0
                        bar_cls = "sector-bar-pos" if ret >= 0 else "sector-bar-neg"
                        val_cls = "sector-val-pos" if ret >= 0 else "sector-val-neg"
                        sign = "+" if ret >= 0 else ""
                        sec_html += f"<div class='sector-row'><div class='sector-name'>{sector}</div><div class='{bar_cls}' style='width:{bar_w}px'></div><div class='{val_cls}'>{sign}{ret:.1f}%</div></div>"
                    sec_html += "</div>"
                    st.markdown(sec_html, unsafe_allow_html=True)
                else:
                    st.warning("No sector data for this date.")
            except Exception as e:
                st.error(f"Error: {e}")

        # ── Right: Supporting News ────────────────────────────────────────
        # ── Right: Supporting News ────────────────────────────────────────
        with col_right:
            try:
                spike_news_df = pd.read_parquet(f"{DATA_DIR}/spike_news/")
                spike_news_df["article_date"] = pd.to_datetime(spike_news_df["article_date"]).dt.date
                spike_dt_hist = datetime.date.fromisoformat(selected_date)
                news_filtered = spike_news_df[
                    spike_news_df["article_date"].apply(lambda d: abs((d - spike_dt_hist).days) <= 3)
                ].copy()
                mask = (
                    (news_filtered["title"].str.len() > 20) &
                    (~news_filtered["title"].str.contains(r"#|\.com|\.org|\.net", na=False)) &
                    (~news_filtered["title"].str.contains(r"[A-Fa-f0-9]{6,}", regex=True, na=False)) &
                    (news_filtered["title"].str.contains(" ", na=False)) &
                    (news_filtered["title"].str.replace(r"[^a-zA-Z]", "", regex=True).str.len() > 15) &
                    (~news_filtered["title"].str.contains(r"^Article [A-Z][a-z]+ [A-Z][a-z]+", regex=True, na=False))
                )
                news_filtered = news_filtered[mask]
                # Apply keyword relevance filter based on spike event label
                news_filtered = news_filtered[news_filtered["title"].apply(lambda t: _passes_news_filter(t, selected_event))].head(6)
                news_html = "<div class='card'><div class='chart-title'>Supporting News — GDELT Archive</div>"
                if len(news_filtered) == 0:
                    news_html += "<p style='color:#888;font-size:13px'>No articles found.</p>"
                else:
                    for _, row in news_filtered.iterrows():
                        news_html += f"""<div class='news-item'>
                            <div class='news-time'>{row['article_date']} · {row['domain']}</div>
                            <span class='tag tag-military'>Geopolitical</span> {row['title']}
                        </div>"""
                news_html += "</div>"
                st.markdown(news_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
    st.caption("Source: GDELT 2016–2026 + S&P 500 yfinance | NYU Big Data Project")


# ══════════════════════════════════════════════════════════════════
# TAB 3: Geo-Risk Q&A (RAG) — UI lives in rag/ui.py
# ══════════════════════════════════════════════════════════════════
with tab3:
    from rag.ui import render_qa
    render_qa()
