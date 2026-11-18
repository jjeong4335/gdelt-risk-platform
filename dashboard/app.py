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
from dashboard.constants import SECTOR_MAP

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Geopolitical Risk Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown('<meta http-equiv="refresh" content="900">', unsafe_allow_html=True)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, .stApp, .stApp p, .stApp div, .stApp input, .stApp label, .stApp li, .stApp td, .stApp th {
        font-family: 'Plus Jakarta Sans', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    .stApp { background: #f4f5fb; }
    .block-container { padding-top: 1rem; max-width: 1560px; }
    .stAppDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    header[data-testid="stHeader"] { height: 0; background: transparent; }

    /* hero banner */
    .hero { background: linear-gradient(120deg, #1e1b4b 0%, #3730a3 45%, #7c3aed 80%, #db2777 120%);
            border-radius: 22px; padding: 28px 32px 26px; color: white; position: relative; overflow: hidden;
            box-shadow: 0 12px 30px rgba(67,56,202,0.25); }
    .hero:after { content: ""; position: absolute; right: -80px; top: -80px; width: 320px; height: 320px;
                  border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%); }
    .hero-kicker { font-size: 12px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: #c7d2fe; }
    .hero-title { font-size: 32px; font-weight: 800; letter-spacing: -0.8px; margin: 6px 0 6px; }
    .hero-sub { font-size: 15px; color: #e0e7ff; max-width: 760px; line-height: 1.5; }
    .hero-pills { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
    .hero-pill { background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.22); color: white;
                 border-radius: 999px; padding: 5px 12px; font-size: 12.5px; font-weight: 600; }
    .live-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #4ade80;
                box-shadow: 0 0 0 3px rgba(74,222,128,0.3); margin-right: 6px; }

    /* tabs as pills */
    .stTabs [role="tablist"] { gap: 6px; border-bottom: none !important; margin: 18px 0 4px; background: white;
        padding: 6px; border-radius: 14px; width: fit-content; box-shadow: 0 1px 3px rgba(30,27,75,0.08); }
    .stTabs [role="tab"] { padding: 8px 18px !important; border-radius: 10px; height: auto; border: none; }
    .stTabs [role="tab"] p { font-size: 14.5px !important; font-weight: 700; color: #64748b; }
    .stTabs [role="tab"][aria-selected="true"] { background: linear-gradient(120deg, #4f46e5, #7c3aed); }
    .stTabs [role="tab"][aria-selected="true"] p { color: white !important; }
    .stTabs [role="tab"] { cursor: pointer; } .stTabs [role="tab"]:hover p { color: #4f46e5; }
    .stTabs .react-aria-SelectionIndicator { display: none !important; }
    [data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

    /* sections */
    .sec { margin: 30px 0 14px; }
    .sec-eyebrow { display: inline-block; font-size: 11.5px; font-weight: 800; text-transform: uppercase;
        letter-spacing: 0.9px; padding: 4px 10px; border-radius: 999px; }
    .c-indigo { background: #e0e7ff; color: #4338ca; } .c-pink { background: #fce7f3; color: #be185d; }
    .c-teal { background: #ccfbf1; color: #0f766e; } .c-orange { background: #ffedd5; color: #c2410c; }
    .c-violet { background: #ede9fe; color: #6d28d9; }
    .sec-title { font-size: 23px; font-weight: 800; color: #1e1b4b; margin-top: 8px; letter-spacing: -0.4px; }
    .sec-what { font-size: 14.5px; color: #64748b; margin-top: 4px; max-width: 920px; line-height: 1.55; }

    /* cards */
    .card, .event-card, .step { background: white; border-radius: 18px; padding: 20px 22px;
        box-shadow: 0 1px 3px rgba(30,27,75,0.06), 0 8px 24px rgba(30,27,75,0.05); margin-bottom: 16px; border: none; }
    div[class*="st-key-card_"] { background: white; border: none !important; border-radius: 18px;
        padding: 20px 22px 8px !important; box-shadow: 0 1px 3px rgba(30,27,75,0.06), 0 8px 24px rgba(30,27,75,0.05);
        margin-bottom: 16px; }
    .card-h { font-size: 16px; font-weight: 700; color: #1e1b4b; }
    .card-sub { font-size: 12.5px; color: #94a3b8; margin: 2px 0 12px; }
    .muted { color: #94a3b8; font-size: 13.5px; }
    .pos { color: #059669; } .neg { color: #e11d48; }

    /* KPI cards, each with its own colour */
    .kpi { border-radius: 18px; padding: 18px 20px; min-height: 200px; margin-bottom: 14px; color: #1e1b4b;
           position: relative; overflow: hidden; box-shadow: 0 8px 24px rgba(30,27,75,0.06); }
    .kpi:before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 5px; }
    .k-indigo { background: linear-gradient(160deg, #eef2ff, #ffffff 60%); } .k-indigo:before { background: linear-gradient(90deg,#4f46e5,#818cf8); }
    .k-risk { background: linear-gradient(160deg, #fdf2f8, #ffffff 60%); } .k-risk:before { background: linear-gradient(90deg,#db2777,#f472b6); }
    .k-teal { background: linear-gradient(160deg, #f0fdfa, #ffffff 60%); } .k-teal:before { background: linear-gradient(90deg,#0d9488,#2dd4bf); }
    .k-orange { background: linear-gradient(160deg, #fff7ed, #ffffff 60%); } .k-orange:before { background: linear-gradient(90deg,#ea580c,#fb923c); }
    .kpi-label { font-size: 13px; color: #475569; font-weight: 700; }
    .kpi-value { font-size: 36px; font-weight: 800; color: #1e1b4b; margin: 8px 0 2px; letter-spacing: -1px; }
    .kpi-unit { font-size: 16px; color: #94a3b8; font-weight: 600; margin-left: 2px; }
    .kpi-note { font-size: 13px; color: #334155; font-weight: 500; }
    .kpi-help { font-size: 12px; color: #64748b; margin-top: 12px; padding-top: 10px; border-top: 1px dashed #e2e8f0; line-height: 1.45; }
    .risk-pill { display: inline-block; font-size: 22px; font-weight: 800; padding: 4px 16px; border-radius: 999px; margin: 6px 0 4px; }
    .risk-high { background: #ffe4e6; color: #e11d48; } .risk-medium { background: #fef3c7; color: #d97706; }
    .risk-low { background: #d1fae5; color: #059669; }

    /* news */
    .news { padding: 11px 0; border-bottom: 1px solid #f1f5f9; font-size: 14px; color: #1e293b; line-height: 1.45; font-weight: 500; }
    .news:last-child { border-bottom: none; }
    .news a { color: #1e293b; text-decoration: none; } .news a:hover { color: #4f46e5; }
    .news-meta { font-size: 12px; color: #94a3b8; margin-bottom: 3px; font-weight: 500; }
    .dayb { margin-left: 8px; background: #ede9fe; color: #6d28d9; border-radius: 999px; padding: 1px 8px; font-size: 11px; font-weight: 700; }
    .tag { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 700; margin-right: 8px; }
    .tag-red { background: #ffe4e6; color: #e11d48; } .tag-amber { background: #fef3c7; color: #b45309; }
    .tag-blue { background: #e0e7ff; color: #4338ca; } .tag-purple { background: #f3e8ff; color: #7e22ce; }
    .tag-orange { background: #ffedd5; color: #c2410c; } .tag-grey { background: #f1f5f9; color: #475569; }

    /* sector bars */
    .bar-row { display: grid; grid-template-columns: 130px 1fr 64px; align-items: center; gap: 10px; margin: 10px 0; font-size: 14px; }
    .bar-name { color: #334155; font-weight: 600; } .bar-track { background: #f1f5f9; border-radius: 999px; height: 10px; }
    .bar { height: 10px; border-radius: 999px; } .bar-pos { background: linear-gradient(90deg,#34d399,#059669); }
    .bar-neg { background: linear-gradient(90deg,#fb7185,#e11d48); } .bar-val { text-align: right; font-weight: 700; }

    /* tables */
    .tbl { width: 100%; border-collapse: collapse; font-size: 14px; border: none; }
    .tbl th { text-align: left; font-size: 12px; color: #94a3b8; font-weight: 700; padding: 6px 6px; border: none; border-bottom: 2px solid #f1f5f9; }
    .tbl td { padding: 9px 6px; border: none; border-bottom: 1px solid #f8fafc; color: #334155; }
    .summary { display: flex; justify-content: space-between; font-size: 14px; color: #64748b; padding: 9px 6px 0; }
    .summary b { color: #1e1b4b; }

    /* history stat tiles */
    .stat { border-radius: 16px; padding: 14px 18px; margin-bottom: 14px; color: white; }
    .s1 { background: linear-gradient(135deg,#4f46e5,#7c3aed); } .s2 { background: linear-gradient(135deg,#db2777,#f43f5e); }
    .s3 { background: linear-gradient(135deg,#ea580c,#f59e0b); } .s4 { background: linear-gradient(135deg,#0d9488,#06b6d4); }
    .stat-v { font-size: 26px; font-weight: 800; letter-spacing: -0.5px; } .stat-l { font-size: 12.5px; opacity: 0.9; font-weight: 600; }

    /* event card */
    .event-card { padding: 0; overflow: hidden; }
    .event-top { background: linear-gradient(120deg, #1e1b4b, #4338ca 60%, #7c3aed); color: white; padding: 18px 22px; }
    .event-name { font-size: 24px; font-weight: 800; letter-spacing: -0.5px; }
    .badge { font-size: 12.5px; font-weight: 700; color: #1e1b4b; background: #fde68a; border-radius: 999px; padding: 3px 10px; margin-left: 10px; vertical-align: middle; }
    .event-facts { display: flex; gap: 36px; margin-top: 12px; flex-wrap: wrap; }
    .event-facts span { display: block; font-size: 12px; color: #c7d2fe; font-weight: 600; } .event-facts b { font-size: 17px; color: white; }
    .event-body { padding: 14px 22px 16px; }
    .event-row { margin-top: 6px; font-size: 13px; } .event-k { color: #64748b; margin-right: 10px; font-weight: 600; }
    .chip { display: inline-block; background: #f3e8ff; color: #7e22ce; border-radius: 999px; padding: 3px 11px; margin: 3px 6px 3px 0; font-size: 12.5px; font-weight: 600; }
    .chip-blue { background: #e0f2fe; color: #0369a1; } .chip-grey { background: #f1f5f9; color: #475569; }

    /* how it works */
    .step { min-height: 300px; } .step-n { width: 36px; height: 36px; border-radius: 12px; color: white;
        font-weight: 800; display: flex; align-items: center; justify-content: center; font-size: 16px; }
    .n1 { background: linear-gradient(135deg,#4f46e5,#818cf8); } .n2 { background: linear-gradient(135deg,#db2777,#f472b6); }
    .n3 { background: linear-gradient(135deg,#7c3aed,#a78bfa); } .n4 { background: linear-gradient(135deg,#0d9488,#2dd4bf); }
    .n5 { background: linear-gradient(135deg,#ea580c,#fb923c); }
    .step-t { font-size: 16px; font-weight: 800; color: #1e1b4b; margin: 12px 0 6px; } .step-b { font-size: 13.5px; color: #64748b; line-height: 1.6; }
    .limits { font-size: 14px; color: #475569; line-height: 1.75; margin: 6px 0 0; }
    .footer { text-align: center; font-size: 12.5px; color: #94a3b8; margin: 40px 0 10px; }

    /* chat panel on the right: stays in view while the page scrolls */
    div[data-testid="stColumn"]:has(.st-key-chat_panel) { position: sticky; top: 1rem; align-self: flex-start; }
    .st-key-chat_panel { background: white; border-radius: 20px; box-shadow: 0 12px 32px rgba(30,27,75,0.12);
                         padding: 18px 18px 8px; max-height: calc(100vh - 2rem); overflow-y: auto;
                         border-top: 5px solid #7c3aed; }
    /* floating "Ask the data" button when the panel is closed */
    .st-key-chat_fab { position: fixed; right: 28px; bottom: 28px; z-index: 1000; width: auto; }
    .st-key-chat_fab button { background: linear-gradient(120deg,#4f46e5,#7c3aed,#db2777); color: white; border: none;
        border-radius: 999px; padding: 12px 22px; font-weight: 700; box-shadow: 0 10px 24px rgba(124,58,237,0.4); }
    .st-key-chat_fab button p { color: white; font-weight: 700; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

# Data folder: GEO_DATA_DIR if set, else ~/dashboard_data_rebuilt (written by
# the data_collection/ steps, see the README).
DATA_DIR = os.path.expanduser(os.environ.get("GEO_DATA_DIR") or "~/dashboard_data_rebuilt")
# Without the daily history the parts that need it say so instead of crashing.
HAS_HISTORY = os.path.exists(f"{DATA_DIR}/geo_tension_index.parquet")

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
    """Events detected and named from the data (data_collection/build_events.py)."""
    df = pd.read_parquet(f"{DATA_DIR}/spike_events.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
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
    # Same rule as spike detection: z-score of the uncapped raw score versus
    # this year's days (the 0-10 index is capped, so a threshold on it can
    # land above 10 and never fire).
    z_now = float(latest["z_score"]) if "z_score" in latest else float("nan")
    if z_now > 3:
        risk_level, risk_class = "High", "risk-high"
    elif z_now > 1.5:
        risk_level, risk_class = "Medium", "risk-medium"
    else:
        risk_level, risk_class = "Low", "risk-low"

    events_today = int(latest["total_events"]) if "total_events" in latest else 0
    score_note = f"{'+' if delta_score >= 0 else ''}{delta_score:.1f} since yesterday"
    threshold_note = (f"z-score {z_now:+.1f} vs this year (spike: above 3)"
                      if z_now == z_now else "Re-run build_history for z-scores")
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
# PAGE LAYOUT
# ══════════════════════════════════════════════════════════════════
import datetime as _dt
import pytz
import html as _html

BLUE, AMBER, RED, GREEN, GREY = "#4f46e5", "#f59e0b", "#db2777", "#059669", "#94a3b8"


def section(eyebrow, title, what, color="indigo"):
    """Section header: coloured label, title, and one line saying what it shows."""
    st.markdown(f"<div class='sec'><div class='sec-eyebrow c-{color}'>{eyebrow}</div>"
                f"<div class='sec-title'>{title}</div><div class='sec-what'>{what}</div></div>",
                unsafe_allow_html=True)


def style_fig(fig, height, ytitle=None, y2title=None):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Plus Jakarta Sans, Segoe UI, Roboto, sans-serif", size=12, color="#334155"),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=12)), hovermode="x unified",
        xaxis=dict(showgrid=False, linecolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False, title=ytitle))
    if y2title:
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title=y2title))
    return fig


def humanize_theme(t):
    t = t.replace("TAX_TERROR_GROUP_", "").replace("TAX_RELIGION_", "").replace("TAX_MILITARY_TITLE_", "")
    t = t.replace("TAX_", "").replace("_", " ").lower()
    return t


def chips(items, cls="chip"):
    return "".join(f"<span class='{cls}'>{_html.escape(str(x))}</span>" for x in items if x)


latest_day = geo_tension["date"].max().strftime("%b %d, %Y") if HAS_HISTORY else "latest 15 minutes"
n_events = len(spike_events)

# ── Chat panel on the right; the page narrows when it is open ────
if "chat_open" not in st.session_state:
    st.session_state.chat_open = True
if st.session_state.chat_open:
    main_col, chat_col = st.columns([2.5, 1], gap="large")
else:
    main_col, chat_col = st.container(), None

with main_col:
    st.markdown(
        "<div class='hero'><div class='hero-kicker'>Geopolitical Risk Monitor</div>"
        "<div class='hero-title'>When the world gets tense, how do markets move?</div>"
        "<div class='hero-sub'>A daily tension score from 10 years of global news, every spike found "
        "automatically, and how S&amp;P 500 sectors moved after each one.</div>"
        "<div class='hero-pills'>"
        f"<span class='hero-pill'><span class='live-dot'></span>Data through {latest_day}</span>"
        f"<span class='hero-pill'>{n_events} events detected</span>"
        "<span class='hero-pill'>GDELT news · S&amp;P 500 prices</span>"
        "<span class='hero-pill'>No hand-picked events</span></div></div>", unsafe_allow_html=True)
    tab_live, tab_hist, tab_how = st.tabs(["Overview", "Event history", "How it works"])

if chat_col is None:
    with st.container(key="chat_fab"):
        if st.button("Ask the data", icon=":material/forum:"):
            st.session_state.chat_open = True
            st.rerun()

if chat_col is not None:
    from rag.ui import render_chat
    with chat_col:
        render_chat()

# ══════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab_live:
    section("At a glance", f"Geopolitical tension · {latest_day}",
            "Today's level of international tension in the news, measured the same way as the "
            "10-year history, and what past spikes meant for a sample portfolio.")

    def kpi(col, label, value, note, value_cls="kpi-value", help_text="", color="indigo"):
        col.markdown(f"<div class='kpi k-{color}'><div class='kpi-label'>{label}</div>"
                     f"<div class='{value_cls}'>{value}</div><div class='kpi-note'>{note}</div>"
                     f"<div class='kpi-help'>{help_text}</div></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Geo-Tension Index", "—" if current_score != current_score else f"{current_score:.1f}<span class='kpi-unit'>/10</span>",
        score_note, help_text="Negative tone × volume of international geopolitical news, scaled 0–10.", color="indigo")
    kpi(c2, "Risk level", f"<span class='risk-pill {risk_class}'>{risk_level}</span>", threshold_note,
        value_cls="", help_text="High = more than 3 standard deviations above this year's average (a spike).",
        color="risk")
    kpi(c3, "Geopolitical articles", f"{events_today:,}", events_note,
        help_text="International articles on conflict, sanctions, unrest, diplomacy and more.", color="teal")
    exp_cls = "pos" if weighted_5d >= 0 else "neg"
    kpi(c4, "Sample portfolio after a spike", f"<span class='{exp_cls}'>{'+' if weighted_5d >= 0 else ''}{weighted_5d:.1f}%</span>",
        "average 5 days after past spikes",
        help_text="30% AAPL, 50% XOM, 20% LMT, averaged over all detected events.", color="orange")

    section("Today", "Live news and market fear",
            "The tension score from GDELT's 15-minute feed next to the VIX, and the latest "
            "geopolitical headlines. Refreshes every 15 minutes.", "pink")
    col_chart, col_news = st.columns([1.15, 1], gap="medium")

    with col_chart:
        et = pytz.timezone("America/New_York")
        now_et = datetime.now(et)
        market_open = now_et.weekday() < 5 and (now_et.hour, now_et.minute) >= (9, 30) and now_et.hour < 16
        try:
            import sqlite3
            conn = sqlite3.connect(f"{DATA_DIR}/live_tension.db")
            live_df = pd.read_sql("SELECT timestamp, tension_score FROM live_tension "
                                  "WHERE date(timestamp) >= date('now') ORDER BY timestamp",
                                  conn, parse_dates=["timestamp"])
            conn.close()
        except Exception:
            live_df = pd.DataFrame(columns=["timestamp", "tension_score"])
        fig = go.Figure()
        if len(live_df) > 0:
            fig.add_trace(go.Scatter(x=live_df["timestamp"], y=live_df["tension_score"], name="Tension (live)",
                                     mode="lines+markers", line=dict(color=BLUE, width=2),
                                     fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"))
            chart_note = f"{len(live_df)} readings today"
        elif HAS_HISTORY:
            recent = geo_tension[geo_tension["date"] >= geo_tension["date"].max() - pd.Timedelta(days=30)]
            fig.add_trace(go.Scatter(x=recent["date"], y=recent["geo_tension_index"], name="Tension (daily)",
                                     mode="lines+markers", line=dict(color=BLUE, width=2),
                                     fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"))
            chart_note = "Last 30 days (no live collector running)"
            if len(vix_df) > 0:
                v = vix_df[vix_df["date"] >= recent["date"].min()]
                fig.add_trace(go.Scatter(x=v["date"], y=v["vix"], name="VIX", yaxis="y2",
                                         mode="lines", line=dict(color=AMBER, width=2, dash="dot")))
        else:
            chart_note = ""
        with st.container(border=True, key="card_live_chart"):
            st.markdown(f"<div class='card-h'>Tension vs VIX</div><div class='card-sub'>{chart_note}</div>",
                        unsafe_allow_html=True)
            if len(fig.data) == 0:
                st.markdown("<p class='muted'>No readings yet. They appear once GDELT and market data load.</p>",
                            unsafe_allow_html=True)
            else:
                st.plotly_chart(style_fig(fig, 260, "Tension (0–10)", "VIX"), width="stretch",
                                config={"displayModeBar": False})

    with col_news:
        tag_cls = {"Military": "tag-red", "Sanctions": "tag-amber", "Diplomatic": "tag-blue",
                   "Nuclear": "tag-purple", "Political": "tag-orange"}
        rows = ""
        nd = news_df
        if len(nd) > 0:
            nd = nd[(nd["title"].str.replace(r"[^a-zA-Z]", "", regex=True).str.len() > 15)
                    & ~nd["title"].str.contains(r"#|\.com|\.org|\.net|[A-Fa-f0-9]{6,}", regex=True, na=False)]
        for _, r in nd.head(6).iterrows():
            rows += (f"<div class='news'><div class='news-meta'>{r['time']} · {_html.escape(str(r['source']))}</div>"
                     f"<span class='tag {tag_cls.get(r['category'], 'tag-grey')}'>{r['category']}</span>"
                     f"{_html.escape(str(r['title']))}</div>")
        if not rows:
            rows = "<p class='muted'>The live GDELT feed is unavailable right now.</p>"
        st.markdown(f"<div class='card'><div class='card-h'>Latest geopolitical headlines</div>"
                    f"<div class='card-sub'>From the newest GDELT file · titles come from article URLs</div>"
                    f"{rows}</div>", unsafe_allow_html=True)

    section("Markets", "How sectors are trading",
            "Live moves of the six sector groups this project tracks, and the sample portfolio's "
            "history after past spikes.", "teal")
    col_sec, col_port = st.columns([1.15, 1], gap="medium")
    with col_sec:
        sd = sector_df.dropna(subset=["avg"])
        if len(sd):
            m = sd["avg"].abs().max() or 1
            bars = "".join(
                f"<div class='bar-row'><div class='bar-name'>{r['sector']}</div>"
                f"<div class='bar-track'><div class='bar {'bar-pos' if r['avg'] >= 0 else 'bar-neg'}' "
                f"style='width:{abs(r['avg']) / m * 100:.0f}%'></div></div>"
                f"<div class='bar-val {'pos' if r['avg'] >= 0 else 'neg'}'>{r['avg']:+.1f}%</div></div>"
                for _, r in sd.iterrows())
        else:
            bars = "<p class='muted'>Live ETF prices are unavailable right now.</p>"
        sub = "Today, since the open" if market_open else "Last trading day"
        st.markdown(f"<div class='card'><div class='card-h'>Sector ETFs</div>"
                    f"<div class='card-sub'>{sub} · JETS, SOXX, XLE, ITA, XLK, GLD</div>{bars}</div>",
                    unsafe_allow_html=True)
    with col_port:
        prow = "".join(
            f"<tr><td><b>{r.ticker}</b></td><td>{r.sector}</td><td>{r.weight}%</td>"
            f"<td class='{'pos' if r.r5 >= 0 else 'neg'}'>{r.r5:+.2f}%</td>"
            f"<td class='neg'>{r.wd:+.2f}%</td></tr>" for r in port_df.itertuples())
        st.markdown(f"<div class='card'><div class='card-h'>Sample portfolio, after past spikes</div>"
                    f"<div class='card-sub'>Average over {n_events} detected events</div>"
                    f"<table class='tbl'><tr><th>Ticker</th><th>Sector</th><th>Weight</th>"
                    f"<th>Day +5</th><th>Worst day</th></tr>{prow}</table>"
                    f"<div class='summary'><span>Weighted Day +5</span>"
                    f"<b class='{'pos' if weighted_5d >= 0 else 'neg'}'>{weighted_5d:+.2f}%</b></div>"
                    f"<div class='summary'><span>Most exposed</span><b>{most_exposed}</b></div></div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# EVENT HISTORY
# ══════════════════════════════════════════════════════════════════
with tab_hist:
    years = spike_events["date"].dt.year
    section("10 years", "Geopolitical tension since 2016",
            f"Each point is one day of international news. Pink dots are the {n_events} spike events "
            f"detected automatically ({years.min()}–{years.max()}); hover a dot for its name.", "violet")
    s1, s2, s3, s4 = st.columns(4)
    for k, (col, label, val) in enumerate([(s1, "Days analysed", f"{len(geo_tension):,}" if HAS_HISTORY else "—"),
                            (s2, "Spike events", f"{n_events}"),
                            (s3, "Largest spike (z-score)", f"{spike_events['z_score'].max():.1f}"),
                            (s4, "Years covered", f"{years.min()}–{years.max()}")], 1):
        col.markdown(f"<div class='stat s{k}'><div class='stat-v'>{val}</div><div class='stat-l'>{label}</div></div>",
                     unsafe_allow_html=True)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=geo_tension["date"], y=geo_tension["geo_tension_index"], name="Daily index",
                                  mode="lines", line=dict(color=BLUE, width=1.2),
                                  fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
                                  hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>"))
    fig_hist.add_trace(go.Scatter(x=spike_events["date"], y=spike_events["geo_tension_index"], name="Detected event",
                                  mode="markers", marker=dict(color=RED, size=9, line=dict(color="white", width=1.5)),
                                  text=spike_events["label"], customdata=spike_events["z_score"],
                                  hovertemplate="<b>%{text}</b><br>%{x|%Y-%m-%d} · z-score %{customdata:.1f}<extra></extra>"))
    style_fig(fig_hist, 380, "Geo-Tension Index (0–10)")
    fig_hist.update_layout(hovermode="closest")
    with st.container(border=True, key="card_hist_chart"):
        st.plotly_chart(fig_hist, width="stretch", config={"displayModeBar": False})

    section("All events", "Detected events",
            "Every spike found in the data, newest first. Names are generated from the countries and "
            "news themes that stood out, not written by hand. z-score = how far above that year's "
            "average the day was (spike rule: above 3).", "pink")
    table = spike_events.sort_values("date", ascending=False).assign(
        Date=lambda d: d["date"].dt.date, Event=lambda d: d["label"],
        **{"z-score": lambda d: d["z_score"], "Index": lambda d: d["geo_tension_index"],
           "Spike days": lambda d: d["spike_days"], "Countries": lambda d: d["top_countries"]})
    st.dataframe(table[["Date", "Event", "z-score", "Index", "Spike days", "Countries"]],
                 hide_index=True, width="stretch", height=280,
                 column_config={"z-score": st.column_config.ProgressColumn(
                                    "z-score", format="%.1f", min_value=0,
                                    max_value=float(spike_events["z_score"].max())),
                                "Index": st.column_config.NumberColumn(format="%.1f")})

    section("Deep dive", "What happened, and how markets moved",
            "Pick an event to see its tension around the peak, how each sector moved afterwards, "
            "and the news coverage it came from.", "orange")
    opts = spike_events.sort_values("date", ascending=False)
    labels = [f"{r.date_str} · {r.label}" for r in opts.itertuples()]
    choice = st.selectbox("Event", labels, label_visibility="collapsed")
    ev = opts.iloc[labels.index(choice)]
    sel_date = ev["date_str"]
    sel_d = ev["date"].date()

    window = (f"{pd.Timestamp(ev['start_date']).date()} → {pd.Timestamp(ev['end_date']).date()}"
              if "start_date" in ev else sel_date)
    themes = [humanize_theme(t) for t in str(ev.get("top_themes") or "").split(", ") if t][:6]
    countries = [c for c in str(ev.get("top_countries") or "").split(", ") if c]
    st.markdown(
        f"<div class='event-card'><div class='event-top'><div class='event-name'>{_html.escape(ev['label'])}"
        f"<span class='badge'>{sel_date}</span></div>"
        f"<div class='event-facts'>"
        f"<div><span>z-score</span><b>{ev['z_score']:.1f}</b></div>"
        f"<div><span>Index</span><b>{ev['geo_tension_index']:.1f}/10</b></div>"
        f"<div><span>Elevated</span><b>{window}</b></div>"
        f"<div><span>Spike days</span><b>{int(ev.get('spike_days', 1))}</b></div></div></div><div class='event-body'>"
        f"<div class='event-row'><span class='event-k'>Countries that stood out</span>{chips(countries, 'chip chip-blue')}</div>"
        f"<div class='event-row'><span class='event-k'>Most mentioned overall</span>"
        f"{chips([c for c in str(ev.get('most_mentioned') or '').split(', ') if c], 'chip chip-grey')}</div>"
        f"<div class='event-row'><span class='event-k'>Distinctive news themes</span>{chips(themes)}</div>"
        f"</div></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="medium")
    with col_l:
        start_dt, end_dt = ev["date"] - pd.Timedelta(days=10), ev["date"] + pd.Timedelta(days=10)
        gw = geo_tension[(geo_tension["date"] >= start_dt) & (geo_tension["date"] <= end_dt)]
        vw = vix_df[(vix_df["date"] >= start_dt) & (vix_df["date"] <= end_dt)] if len(vix_df) else vix_df
        fe = go.Figure()
        fe.add_trace(go.Scatter(x=gw["date"], y=gw["geo_tension_index"], name="Tension", mode="lines+markers",
                                line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"))
        if len(vw):
            fe.add_trace(go.Scatter(x=vw["date"], y=vw["vix"], name="VIX", yaxis="y2", mode="lines+markers",
                                    line=dict(color=AMBER, width=2, dash="dot")))
        fe.add_vline(x=ev["date"], line=dict(color=RED, width=1, dash="dash"))
        with st.container(border=True, key="card_event_chart"):
            st.markdown("<div class='card-h'>Tension and VIX around the peak</div>"
                        "<div class='card-sub'>10 days either side · dashed line = peak day</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(style_fig(fe, 280, "Tension", "VIX" if len(vw) else None), width="stretch",
                            config={"displayModeBar": False})

    with col_r:
        reac = load_ticker_reaction_by_spike()
        sp = reac[reac["spike_date"] == sel_d].copy()
        t2s = {t: s for s, ts in SECTOR_MAP.items() for t in ts}
        sp["sector"] = sp["ticker"].map(t2s)
        sp = sp.dropna(subset=["sector"])
        fs = go.Figure()
        base_note = ""
        if len(sp):
            days = sorted(d for d in sp["days_from_spike"].unique() if d > 0)
            colors = {1: "#38bdf8", 5: "#6366f1", 30: "#db2777"}
            for h in (1, 5, 30):
                if not days:
                    break
                used = min(days, key=lambda x: abs(x - h))
                g = sp[sp["days_from_spike"] == used].groupby("sector")["avg_return_pct"].mean()
                g = g.reindex(list(SECTOR_MAP))
                fs.add_trace(go.Bar(x=g.index, y=g.values, name=f"Day +{h}", marker_color=colors[h],
                                    hovertemplate="%{x}: %{y:+.2f}%<extra>Day +" + str(h) + "</extra>"))
            if "base_date" in sp and pd.notna(sp["base_date"].iloc[0]) and pd.Timestamp(sp["base_date"].iloc[0]).date() != sel_d:
                base_note = f" · not a trading day, measured from the {pd.Timestamp(sp['base_date'].iloc[0]).date()} close"
        style_fig(fs, 280, "Average return (%)")
        fs.update_layout(barmode="group", hovermode="closest")
        fs.add_hline(y=0, line=dict(color="#cbd5e1", width=1))
        with st.container(border=True, key="card_sector_chart"):
            st.markdown("<div class='card-h'>Sector returns after the event</div>"
                        f"<div class='card-sub'>Average of S&P 500 members in each group · calendar days after the peak{base_note}</div>",
                        unsafe_allow_html=True)
            if len(fs.data):
                st.plotly_chart(fs, width="stretch", config={"displayModeBar": False})
            else:
                st.markdown("<p class='muted'>No price data for this event.</p>", unsafe_allow_html=True)

    try:
        sn = pd.read_parquet(f"{DATA_DIR}/spike_news/")
        sn["spike_date"] = pd.to_datetime(sn["spike_date"]).dt.date
        sn = sn[sn["spike_date"] == sel_d].copy()
        t = sn["title"].fillna("")
        sn = sn[(t.str.len() > 20) & t.str.contains(" ") & ~t.str.contains(r"%|#|\.com|\.org|\.net|[A-Fa-f0-9]{6,}", regex=True)
                & (t.str.replace(r"[^a-zA-Z]", "", regex=True).str.len() > 15)]
        sn = sn.assign(dist=sn["days_from_spike"].abs()).sort_values("dist").drop_duplicates("title").head(8)
        items = "".join(
            f"<div class='news'><div class='news-meta'>{r.article_date} · {_html.escape(str(r.domain))}"
            f"<span class='dayb'>{'peak day' if r.days_from_spike == 0 else f'{r.days_from_spike:+d} d'}</span></div>"
            f"<a href='{_html.escape(str(r.url))}' target='_blank'>{_html.escape(r.title)}</a></div>"
            for r in sn.itertuples()) or "<p class='muted'>No readable headlines for this event.</p>"
    except Exception:
        items = "<p class='muted'>News for this event isn't available.</p>"
    st.markdown(f"<div class='card'><div class='card-h'>News coverage</div>"
                f"<div class='card-sub'>Articles from 3 days before to 3 days after the peak, closest first · "
                f"titles come from article URLs</div>{items}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HOW IT WORKS
# ══════════════════════════════════════════════════════════════════
with tab_how:
    section("Method", "How the numbers are made",
            "Everything on this site is computed from public data. Nothing is labelled by hand, "
            "and nothing here is a prediction or investment advice.")
    steps = [
        ("1", "Measure tension", "Every day, take GDELT's international news on conflict, sanctions, unrest, "
         "diplomacy, cyber and nuclear topics. Score = average negative tone × log(number of articles), "
         "shown on a 0–10 scale."),
        ("2", "Detect spikes", "A day is a spike when its score is more than 3 standard deviations above that "
         "year's average. Spike days within 3 days of each other are one event, dated by its peak."),
        ("3", "Name events", "Each event is named after the countries and news categories that appear far "
         "more often than around other events (a lift score, like TF-IDF). No AI or hand labels."),
        ("4", "Measure markets", "For every event, the return of each S&P 500 stock from the first trading "
         "day on or after the peak, averaged over six sector groups."),
        ("5", "Ask the data", "The assistant on the right searches these results and answers only from them, "
         "with numbered sources. If the data has no answer, it says so."),
    ]
    cols = st.columns(len(steps))
    for col, (n, title, body) in zip(cols, steps):
        col.markdown(f"<div class='step'><div class='step-n n{n}'>{n}</div><div class='step-t'>{title}</div>"
                     f"<div class='step-b'>{body}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='card' style='margin-top:18px'><div class='card-h'>Limits worth knowing</div>"
                "<ul class='limits'><li>Returns are averages that include all other market news in the window: "
                "correlation, not causation.</li><li>Stock list = today's S&P 500, so companies that left the "
                "index are missing.</li><li>Event names are coarse (country + category) and can miss the "
                "specifics.</li><li>Headlines are rebuilt from article URLs, not full text.</li></ul></div>",
                unsafe_allow_html=True)

st.markdown("<div class='footer'>Data: GDELT 2.0 (via BigQuery) and S&amp;P 500 prices from yfinance · "
            "Started as an NYU Big Data project · Not investment advice</div>", unsafe_allow_html=True)
