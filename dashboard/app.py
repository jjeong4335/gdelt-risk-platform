import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import zipfile
import io
from datetime import datetime
import yfinance as yf

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
    .sector-bar-pos { background: #fca5a5; height: 8px; border-radius: 4px; }
    .sector-bar-neg { background: #86efac; height: 8px; border-radius: 4px; }
    .sector-val-pos { color: #dc2626; font-size: 13px; font-weight: 600; margin-left: 8px; }
    .sector-val-neg { color: #16a34a; font-size: 13px; font-weight: 600; margin-left: 8px; }
    .port-row { display: flex; align-items: center; padding: 6px 0; font-size: 13px; }
    .port-ticker { font-weight: 700; width: 50px; }
    .port-sector { color: #888; width: 90px; }
    .port-weight { color: #444; width: 40px; }
    .port-return-pos { color: #dc2626; font-weight: 600; margin-left: auto; }
    .port-return-neg { color: #16a34a; font-weight: 600; margin-left: auto; }
    .summary-row { display: flex; justify-content: space-between; padding: 6px 0; font-size: 13px; border-top: 1px solid #f0f0f0; }
    .summary-label { color: #666; }
    .summary-val-pos { color: #16a34a; font-weight: 600; }
    .summary-val-neg { color: #dc2626; font-weight: 600; }
    .chart-title { font-size: 13px; color: #444; font-weight: 500; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Sector mapping ────────────────────────────────────────────────
SECTOR_MAP = {
    "Defense":         ["LMT", "RTX", "NOC", "GD", "HII", "BA", "TDG", "LHX"],
    "Energy":          ["XOM", "CVX", "COP", "OXY", "SLB", "EOG", "PSX", "VLO", "MPC"],
    "Gold/Safe Haven": ["NEM", "FCX", "KO", "PG", "JNJ", "CL", "GIS"],
    "Tech":            ["AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN", "NVDA", "ORCL", "CRM"],
    "Semiconductors":  ["AMD", "INTC", "QCOM", "NXPI", "AMAT", "LRCX", "MCHP", "TXN"],
    "Airlines":        ["AAL", "UAL", "DAL", "LUV", "NCLH", "RCL", "CCL"],
}

DATA_DIR = "/home/jj4335_nyu_edu/dashboard_data"

# ── Spike event labels ────────────────────────────────────────────
SPIKE_LABELS = {
    "2016-07-08": "Dallas Police Shooting",
    "2016-06-23": "Brexit Referendum",
    "2016-11-10": "Trump Election",
    "2016-11-12": "Trump Election Aftermath",
    "2017-01-20": "Trump Inauguration",
    "2018-01-02": "North Korea Tensions",
    "2019-07-05": "Iran Strait of Hormuz",
    "2019-09-19": "Saudi Oil Attack",
    "2020-01-07": "Soleimani Assassination",
    "2020-06-01": "George Floyd Protests",
    "2020-06-08": "BLM Protests Peak",
    "2021-01-07": "US Capitol Riot",
    "2022-02-24": "Russia-Ukraine War",
    "2023-10-07": "Hamas Attack on Israel",
    "2024-04-14": "Iran Attack on Israel",
}

# ── Load data ─────────────────────────────────────────────────────
@st.cache_data
def load_geo_tension():
    df = pd.read_parquet(f"{DATA_DIR}/geo_tension_index.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year.between(2016, 2026)]
    return df.sort_values("date")

@st.cache_data
def load_spike_events():
    df = pd.read_parquet(f"{DATA_DIR}/spike_events.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    df["label"] = df["date_str"].map(SPIKE_LABELS).fillna("Geopolitical Event")
    return df.sort_values("geo_tension_index", ascending=False)

@st.cache_data
def load_ticker_summary():
    return pd.read_parquet(f"{DATA_DIR}/ticker_summary.parquet")

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

@st.cache_data(ttl=900)
def fetch_latest_gdelt_news():
    try:
        resp = requests.get("http://data.gdeltproject.org/gdeltv2/lastupdate.txt", timeout=10)
        lines = resp.text.strip().split("\n")
        gkg_url = [l.split()[-1] for l in lines if "gkg.csv.zip" in l][0]
        r = requests.get(gkg_url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        fname = z.namelist()[0]
        with z.open(fname) as f:
            df = pd.read_csv(f, sep="\t", header=None, on_bad_lines="skip",
                           usecols=[1, 3, 4, 7],
                           names=["date", "source", "url", "themes"])

        # Filter geopolitical keywords
        keywords = "MILITARY|WAR|WEAPON|MISSILE|SANCTION|EMBARGO|COUP|PROTEST|DIPLOMATIC|NUCLEAR|CYBERATTACK"
        df = df[df["themes"].str.contains(keywords, na=False, case=False)]

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
                return title[:60] if title else str(url)[:60]
            except:
                return str(url)[:60]

        df["category"] = df["themes"].apply(get_category)
        df["title"] = df["url"].apply(extract_title)
        df["time"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time", ascending=False)
        return df.head(10)[["time", "title", "source", "category"]].reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame(columns=["time", "title", "source", "category"])

# Load all data
geo_tension = load_geo_tension()
spike_events = load_spike_events()
ticker_summary = load_ticker_summary()
vix_df = fetch_vix()
news_df = fetch_latest_gdelt_news()

# ── Compute metrics ───────────────────────────────────────────────
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

sector_rows = []
for sector, tickers in SECTOR_MAP.items():
    subset = ticker_summary[ticker_summary["ticker"].isin(tickers)]
    if len(subset) > 0:
        sector_rows.append({"sector": sector, "avg": subset["avg_return_5d"].mean()})
sector_df = pd.DataFrame(sector_rows).sort_values("avg", ascending=False)

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
tab1, tab2 = st.tabs(["📊 Live Dashboard", "📅 Historical Events"])

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
            <span class='live-badge'>● Live — auto-refreshes every 15 min</span>
        </div>
        """, unsafe_allow_html=True)

    # Top 4 metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        delta_class = "metric-delta-pos" if delta_score >= 0 else "metric-delta-neg"
        sign = "+" if delta_score >= 0 else ""
        st.markdown(f"""<div class='card'>
            <div class='metric-label'>Geo-Tension Index</div>
            <div class='metric-value'>{current_score:.1f}</div>
            <div class='{delta_class}'>{sign}{delta_score:.1f} since yesterday</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class='card'>
            <div class='metric-label'>Risk Level</div>
            <div class='{risk_class}'>{risk_level}</div>
            <div class='metric-delta-neu'>Threshold: {threshold:.1f}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class='card'>
            <div class='metric-label'>Events Today</div>
            <div class='metric-value'>{events_today:,}</div>
            <div class='metric-delta-neu'>Geopolitical news volume</div>
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

        # Geo-Tension Index (full period)
        fig.add_trace(go.Scatter(
            x=geo_tension["date"], y=geo_tension["geo_tension_index"],
            mode="lines", fill="tozeroy", name="Geo-Tension Index",
            line=dict(color="#3b82f6", width=2),
            fillcolor="rgba(59,130,246,0.1)",
            yaxis="y1"
        ))

        # VIX (full period)
        if len(vix_df) > 0:
            fig.add_trace(go.Scatter(
                x=vix_df["date"], y=vix_df["vix"],
                mode="lines", name="VIX",
                line=dict(color="#f59e0b", width=2, dash="dot"),
                yaxis="y2"
            ))

        fig.update_layout(
            height=240,
            margin=dict(l=0, r=40, t=10, b=0),
            xaxis=dict(showgrid=False, tickformat="%Y"),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Tension", side="left"),
            yaxis2=dict(overlaying="y", side="right", title="VIX", showgrid=False),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.1, x=0)
        )

        st.markdown("<div class='card'><div class='chart-title'>Geo-Tension Index vs VIX (2016–2026)</div>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_news:
        tag_classes = {
            "Military": "tag-military", "Sanctions": "tag-sanctions",
            "Diplomatic": "tag-diplomatic", "Nuclear": "tag-nuclear",
            "Political": "tag-political", "Other": "tag-other"
        }
        news_html = "<div class='card'><div class='chart-title'>Latest news events</div>"
        if len(news_df) > 0:
            for _, row in news_df.head(5).iterrows():
                time_str = row["time"].strftime("%H:%M UTC") if pd.notna(row["time"]) else ""
                cat = row["category"]
                tag_cls = tag_classes.get(cat, "tag-other")
                title = str(row["title"]) if pd.notna(row["title"]) else "Unknown"
                source = str(row["source"]) if pd.notna(row["source"]) else ""
                news_html += f"""<div class='news-item'>
                    <div class='news-time'>{time_str} · {source}</div>
                    <span class='tag {tag_cls}'>{cat}</span>{title}
                </div>"""
        else:
            news_html += "<p style='color:#888;font-size:13px'>No recent events found.</p>"
        news_html += "</div>"
        st.markdown(news_html, unsafe_allow_html=True)

    # Sector + Portfolio
    col_sector, col_port = st.columns([1, 1])

    with col_sector:
        max_abs = sector_df["avg"].abs().max()
        sector_html = "<div class='card'><div class='chart-title'>Sector reaction — past similar events (avg 5-day)</div>"
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
        sector_html += "</div>"
        st.markdown(sector_html, unsafe_allow_html=True)

    with col_port:
        port_html = "<div class='card'><div class='chart-title'>Portfolio risk calculator</div>"
        for _, row in port_df.iterrows():
            r5 = row["r5"]
            ret_cls = "port-return-pos" if r5 >= 0 else "port-return-neg"
            sign = "+" if r5 >= 0 else ""
            port_html += f"""<div class='port-row'>
                <div class='port-ticker'>{row['ticker']}</div>
                <div class='port-sector'>{row['sector']}</div>
                <div class='port-weight'>{row['weight']}%</div>
                <div class='{ret_cls}'>{sign}{r5:.1f}%</div>
            </div>"""

        wd_sign = "+" if weighted_5d >= 0 else ""
        most_exp_risk = "risk-high" if risk_level == "High" else "risk-medium" if risk_level == "Medium" else "risk-low"
        port_html += f"""<div style='margin-top:12px'>
            <div class='summary-row'>
                <span class='summary-label'>Hist. avg change</span>
                <span class='summary-val-{"pos" if weighted_5d >= 0 else "neg"}'>{wd_sign}{weighted_5d:.1f}%</span>
            </div>
            <div class='summary-row'>
                <span class='summary-label'>Worst drawdown</span>
                <span class='summary-val-neg'>{weighted_wd:.1f}%</span>
            </div>
            <div class='summary-row'>
                <span class='summary-label'>Most exposed holding</span>
                <span style='font-weight:600'>{most_exposed}</span>
            </div>
            <div class='summary-row'>
                <span class='summary-label'>Current tension level</span>
                <span class='{most_exp_risk}'>{risk_level} ({current_score:.0f})</span>
            </div>
        </div></div>"""
        st.markdown(port_html, unsafe_allow_html=True)

    st.caption("Source: GDELT (live) + S&P 500 2016–2026 | NYU Big Data Project")

# ══════════════════════════════════════════════════════════════════
# TAB 2: HISTORICAL EVENTS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📅 Historical Geopolitical Spike Events")
    st.caption("45 major geopolitical events detected from GDELT news sentiment (2016–2026)")

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
    fig_hist.update_layout(
        height=350,
        xaxis=dict(showgrid=False, title="Date"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", title="Tension Score"),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.05),
        hovermode="closest"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("🚨 Top Spike Events")
    display_df = spike_events[["date", "label", "geo_tension_index", "total_events"]].copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df.columns = ["Date", "Event", "Tension Score", "News Volume"]
    display_df["Tension Score"] = display_df["Tension Score"].round(2)
    st.dataframe(display_df.head(20).reset_index(drop=True), use_container_width=True)

    st.divider()

    st.subheader("📈 Ticker Reaction Patterns During Spike Events")
    col_g, col_l = st.columns(2)

    with col_g:
        st.markdown("**Top 15 Gainers (5-day avg)**")
        top_gainers = ticker_summary.nlargest(15, "avg_return_5d")[["ticker", "avg_return_5d", "worst_drawdown"]]
        fig_g = go.Figure(go.Bar(
            x=top_gainers["avg_return_5d"],
            y=top_gainers["ticker"],
            orientation="h",
            marker_color="#3b82f6",
            text=[f"{v:+.1f}%" for v in top_gainers["avg_return_5d"]],
            textposition="outside"
        ))
        fig_g.update_layout(height=400, margin=dict(l=0, r=60, t=0, b=0),
                           plot_bgcolor="white", paper_bgcolor="white",
                           xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(fig_g, use_container_width=True)

    with col_l:
        st.markdown("**Top 15 Losers (worst drawdown)**")
        top_losers = ticker_summary.nsmallest(15, "worst_drawdown")[["ticker", "avg_return_5d", "worst_drawdown"]]
        fig_l = go.Figure(go.Bar(
            x=top_losers["worst_drawdown"],
            y=top_losers["ticker"],
            orientation="h",
            marker_color="#dc2626",
            text=[f"{v:.1f}%" for v in top_losers["worst_drawdown"]],
            textposition="outside"
        ))
        fig_l.update_layout(height=400, margin=dict(l=0, r=60, t=0, b=0),
                           plot_bgcolor="white", paper_bgcolor="white",
                           xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(fig_l, use_container_width=True)

    st.caption("Source: GDELT 2016–2026 + S&P 500 yfinance | NYU Big Data Project")
