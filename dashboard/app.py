import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import zipfile
import io
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Geopolitical Risk Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ── Sector mapping ────────────────────────────────────────────────
SECTOR_MAP = {
    "Defense":        ["LMT", "RTX", "NOC", "GD", "HII", "BA", "TDG", "LHX"],
    "Energy":         ["XOM", "CVX", "COP", "OXY", "SLB", "EOG", "PSX", "VLO", "MPC", "HES"],
    "Gold/Safe Haven":["NEM", "FCX", "KO", "PG", "JNJ", "CL", "GIS", "K"],
    "Tech":           ["AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN", "NVDA", "ORCL", "CRM", "ADBE"],
    "Semiconductors": ["AMD", "INTC", "QCOM", "NXPI", "AMAT", "LRCX", "MCHP", "KLAC", "ASML", "TXN"],
    "Airlines":       ["AAL", "UAL", "DAL", "LUV", "NCLH", "RCL", "CCL", "ALK", "JBLU"],
}

# ── Data directory ────────────────────────────────────────────────
DATA_DIR = "/home/jj4335_nyu_edu/dashboard_data"

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
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

@st.cache_data
def load_ticker_summary():
    return pd.read_parquet(f"{DATA_DIR}/ticker_summary.parquet")

@st.cache_data(ttl=900)  # refresh every 15 minutes
def fetch_latest_gdelt_news():
    """Fetch latest GDELT news events directly from GDELT API."""
    try:
        # Get last update file list
        resp = requests.get(
            "http://data.gdeltproject.org/gdeltv2/lastupdate.txt",
            timeout=10
        )
        lines = resp.text.strip().split("\n")
        gkg_url = [l.split()[-1] for l in lines if "gkg.csv.zip" in l][0]

        # Download and parse
        r = requests.get(gkg_url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        fname = z.namelist()[0]
        with z.open(fname) as f:
            df = pd.read_csv(f, sep="\t", header=None, on_bad_lines="skip",
                           usecols=[0, 4, 6, 15],
                           names=["date", "source", "themes", "tone"])

        # Filter geopolitical keywords
        keywords = "MILITARY|WAR|WEAPON|MISSILE|SANCTION|EMBARGO|COUP|PROTEST|DIPLOMATIC|NUCLEAR|CYBERATTACK"
        df = df[df["themes"].str.contains(keywords, na=False, case=False)]

        # Parse category
        def get_category(themes):
            if any(k in str(themes) for k in ["MILITARY", "WAR", "WEAPON", "MISSILE", "INVASION"]):
                return "Military"
            elif any(k in str(themes) for k in ["SANCTION", "EMBARGO", "TARIFF"]):
                return "Sanctions"
            elif any(k in str(themes) for k in ["DIPLOMATIC", "EXPULSION"]):
                return "Diplomatic"
            elif any(k in str(themes) for k in ["NUCLEAR",]):
                return "Nuclear"
            elif any(k in str(themes) for k in ["COUP", "PROTEST"]):
                return "Political"
            else:
                return "Other"

        df["category"] = df["themes"].apply(get_category)
        df["time"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time", ascending=False)
        return df.head(20)[["time", "source", "category", "themes"]].reset_index(drop=True)

    except Exception as e:
        return pd.DataFrame(columns=["time", "source", "category", "themes"])

# ── Load all data ─────────────────────────────────────────────────
geo_tension = load_geo_tension()
spike_events = load_spike_events()
ticker_summary = load_ticker_summary()

# ── Compute current metrics ───────────────────────────────────────
latest = geo_tension.iloc[-1]
prev = geo_tension.iloc[-2]
current_score = latest["geo_tension_index"]
delta_score = current_score - prev["geo_tension_index"]
threshold = geo_tension["geo_tension_index"].mean() + 3 * geo_tension["geo_tension_index"].std()

if current_score > threshold:
    risk_level = "High"
    risk_color = "red"
elif current_score > threshold * 0.7:
    risk_level = "Medium"
    risk_color = "orange"
else:
    risk_level = "Low"
    risk_color = "green"

events_today = latest["total_events"] if "total_events" in latest else 0

# ── Compute sector reactions ──────────────────────────────────────
def compute_sector_reactions(ticker_summary, sector_map):
    rows = []
    for sector, tickers in sector_map.items():
        subset = ticker_summary[ticker_summary["ticker"].isin(tickers)]
        if len(subset) > 0:
            avg_5d = subset["avg_return_5d"].mean()
            rows.append({"sector": sector, "avg_return_5d": avg_5d})
    return pd.DataFrame(rows).sort_values("avg_return_5d", ascending=False)

sector_reactions = compute_sector_reactions(ticker_summary, SECTOR_MAP)

# ══════════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ══════════════════════════════════════════════════════════════════

st.title("🌍 Geopolitical Risk Dashboard")
st.caption("VIX tells you how scared the market is. This tells you why — and how exposed your portfolio is.")

# Live indicator
col_live, _ = st.columns([1, 5])
with col_live:
    st.markdown(f"🟢 **Live** — updated {datetime.utcnow().strftime('%H:%M')} UTC")

st.divider()

# ── Top metrics row ───────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Geo-Tension Index",
        f"{current_score:.1f}",
        delta=f"{delta_score:+.1f} since yesterday"
    )

with col2:
    st.markdown(f"**Risk Level**")
    st.markdown(f"<h2 style='color:{risk_color}'>{risk_level}</h2>", unsafe_allow_html=True)
    st.caption(f"Threshold: {threshold:.1f}")

with col3:
    st.metric(
        "Events Today",
        f"{int(events_today):,}",
    )

with col4:
    # Compute default portfolio exposure
    default_portfolio = [("AAPL", 30.0), ("XOM", 50.0), ("LMT", 20.0)]
    df_port = pd.DataFrame(default_portfolio, columns=["ticker", "weight"])
    df_port = df_port.merge(ticker_summary, on="ticker", how="left")
    weighted_5d = (df_port["avg_return_5d"] * df_port["weight"] / 100).sum()
    st.metric("Portfolio Exposure", f"{weighted_5d:+.1f}%",
              delta="Hist. avg on similar events")

st.divider()

# ── Main content: chart + news feed ──────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Geo-Tension Index — last 30 days")
    recent = geo_tension.tail(30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["date"],
        y=recent["geo_tension_index"],
        mode="lines",
        fill="tozeroy",
        line=dict(color="#2E75B6", width=2),
        fillcolor="rgba(46,117,182,0.15)"
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#eee"),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Latest news events")
    with st.spinner("Fetching live GDELT news..."):
        news_df = fetch_latest_gdelt_news()

    if len(news_df) > 0:
        category_colors = {
            "Military": "🔴",
            "Sanctions": "🟡",
            "Diplomatic": "🔵",
            "Nuclear": "⚫",
            "Political": "🟠",
            "Other": "⚪"
        }
        for _, row in news_df.head(6).iterrows():
            time_str = row["time"].strftime("%H:%M UTC") if pd.notna(row["time"]) else ""
            cat = row["category"]
            icon = category_colors.get(cat, "⚪")
            source = str(row["source"])[:50] if pd.notna(row["source"]) else "Unknown"
            st.markdown(f"`{time_str}` {icon} **{cat}** — {source}")
    else:
        st.info("No recent geopolitical news found.")

st.divider()

# ── Bottom section: sector reactions + portfolio calculator ───────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Sector reaction — past similar events (avg 5-day)")

    colors = ["#2E75B6" if v >= 0 else "#C00000" for v in sector_reactions["avg_return_5d"]]
    fig_sector = go.Figure(go.Bar(
        x=sector_reactions["avg_return_5d"],
        y=sector_reactions["sector"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in sector_reactions["avg_return_5d"]],
        textposition="outside"
    ))
    fig_sector.update_layout(
        height=300,
        margin=dict(l=0, r=60, t=10, b=0),
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#ccc"),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    st.plotly_chart(fig_sector, use_container_width=True)

with col_right:
    st.subheader("Portfolio risk calculator")

    available_tickers = sorted(ticker_summary["ticker"].dropna().unique().tolist())
    num_holdings = st.slider("Holdings", 1, 10, 3, label_visibility="collapsed")

    holdings = []
    total_weight = 0
    default_tickers = ["AAPL", "XOM", "LMT"]
    default_weights = [30.0, 50.0, 20.0]

    # Determine sector for each ticker
    def get_sector(ticker):
        for sector, tickers in SECTOR_MAP.items():
            if ticker in tickers:
                return sector
        return "Other"

    for i in range(num_holdings):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            default_idx = available_tickers.index(default_tickers[i]) \
                if i < len(default_tickers) and default_tickers[i] in available_tickers else 0
            ticker = st.selectbox("", options=available_tickers,
                                  key=f"ticker_{i}", index=default_idx,
                                  label_visibility="collapsed")
        with c2:
            st.markdown(f"<small style='color:gray'>{get_sector(ticker)}</small>", unsafe_allow_html=True)
            weight = st.number_input("", min_value=0.0, max_value=100.0,
                                     value=default_weights[i] if i < len(default_weights) else 10.0,
                                     key=f"weight_{i}", label_visibility="collapsed")
        with c3:
            stat = ticker_summary[ticker_summary["ticker"] == ticker]
            if len(stat) > 0:
                r5 = stat.iloc[0]["avg_return_5d"]
                color = "green" if r5 >= 0 else "red"
                st.markdown(f"<p style='color:{color};margin-top:28px'>{r5:+.1f}%</p>",
                           unsafe_allow_html=True)
        holdings.append((ticker, weight))
        total_weight += weight

    st.divider()

    portfolio_df = pd.DataFrame(holdings, columns=["ticker", "weight"])
    portfolio_with_stats = portfolio_df.merge(ticker_summary, on="ticker", how="left")

    weighted_5d = (portfolio_with_stats["avg_return_5d"] * portfolio_with_stats["weight"] / 100).sum()
    weighted_30d = (portfolio_with_stats["avg_return_30d"] * portfolio_with_stats["weight"] / 100).sum()
    weighted_drawdown = (portfolio_with_stats["worst_drawdown"] * portfolio_with_stats["weight"] / 100).sum()
    exposure_scores = portfolio_with_stats["worst_drawdown"].abs() * portfolio_with_stats["weight"] / 100
    most_exposed = portfolio_with_stats.loc[exposure_scores.idxmax(), "ticker"] if len(portfolio_with_stats) > 0 else "N/A"

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Hist. avg change**")
        st.markdown("**Worst drawdown**")
        st.markdown("**Most exposed holding**")
        st.markdown("**Current tension level**")
    with col_b:
        c5d = "green" if weighted_5d >= 0 else "red"
        st.markdown(f"<p style='color:{c5d}'>{weighted_5d:+.1f}%</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:red'>{weighted_drawdown:.1f}%</p>", unsafe_allow_html=True)
        st.markdown(f"**{most_exposed}**")
        st.markdown(f"<p style='color:{risk_color}'>{risk_level} ({current_score:.0f})</p>",
                   unsafe_allow_html=True)

st.divider()
st.caption(f"Data: GDELT (live) + S&P 500 2016–2026 | NYU Big Data Project | Refreshes every 15 min")
