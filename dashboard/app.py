import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Geopolitical Risk Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Geopolitical Event-Driven Portfolio Risk Analysis")
st.caption("VIX tells you how scared the market is. This tells you why — and how exposed your portfolio is.")

DATA_DIR = "/home/jj4335_nyu_edu/dashboard_data"

@st.cache_data
def load_geo_tension():
    df = pd.read_parquet(f"{DATA_DIR}/geo_tension_index.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year.between(2016, 2026)]  # filter valid years
    return df.sort_values("date")

@st.cache_data
def load_spike_events():
    df = pd.read_parquet(f"{DATA_DIR}/spike_events.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.date  # date only, no time
    return df

@st.cache_data
def load_ticker_summary():
    return pd.read_parquet(f"{DATA_DIR}/ticker_summary.parquet")

with st.spinner("Loading data..."):
    geo_tension = load_geo_tension()
    spike_events = load_spike_events()
    ticker_summary = load_ticker_summary()

# ── Section 1: Geo-Tension Index ──────────────────────────────────
st.header("📈 Geo-Tension Index (2016–2026)")

fig_tension = go.Figure()

fig_tension.add_trace(go.Scatter(
    x=geo_tension["date"],
    y=geo_tension["geo_tension_index"],
    mode="lines",
    name="Geo-Tension Index",
    line=dict(color="#2E75B6", width=1.5)
))

fig_tension.add_trace(go.Scatter(
    x=spike_events["date"],
    y=spike_events["geo_tension_index"],
    mode="markers",
    name="Spike Events",
    marker=dict(color="red", size=8, symbol="circle")
))

fig_tension.update_layout(
    height=400,
    xaxis_title="Date",
    yaxis_title="Tension Score",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)

st.plotly_chart(fig_tension, use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Spike Events", len(spike_events))
with col2:
    latest = geo_tension.iloc[-1]
    st.metric("Latest Tension Score", f"{latest['geo_tension_index']:.2f}")
with col3:
    st.metric("Total Trading Days", len(geo_tension))

# ── Section 2: Spike Events Table ─────────────────────────────────
st.header("🚨 Spike Events")
st.dataframe(
    spike_events[["date", "geo_tension_index", "total_events"]] \
        .sort_values("geo_tension_index", ascending=False) \
        .head(20) \
        .reset_index(drop=True),
    use_container_width=True
)

# ── Section 3: Portfolio Risk Calculator ──────────────────────────
st.header("💼 Portfolio Risk Calculator")
st.caption("Enter your portfolio holdings to see historical risk exposure during geopolitical events.")

available_tickers = sorted(ticker_summary["ticker"].dropna().unique().tolist())

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Portfolio Holdings")
    num_holdings = st.slider("Number of holdings", 1, 10, 3)

    holdings = []
    total_weight = 0

    for i in range(num_holdings):
        cols = st.columns([3, 2])
        with cols[0]:
            default_tickers = ["AAPL", "XOM", "LMT"]
            default_idx = available_tickers.index(default_tickers[i]) \
                if i < len(default_tickers) and default_tickers[i] in available_tickers else 0
            ticker = st.selectbox(
                f"Ticker {i+1}",
                options=available_tickers,
                key=f"ticker_{i}",
                index=default_idx
            )
        with cols[1]:
            default_weights = [30.0, 50.0, 20.0]
            weight = st.number_input(
                f"Weight % {i+1}",
                min_value=0.0,
                max_value=100.0,
                value=default_weights[i] if i < len(default_weights) else 10.0,
                key=f"weight_{i}"
            )
        holdings.append((ticker, weight))
        total_weight += weight

with col2:
    st.subheader("Summary")
    st.metric("Total Weight", f"{total_weight:.1f}%",
              delta="OK" if abs(total_weight - 100) < 0.1 else f"{total_weight - 100:.1f}% off")

if st.button("Calculate Risk", type="primary"):
    if abs(total_weight - 100) > 1:
        st.error(f"Total weight must equal 100%. Currently: {total_weight:.1f}%")
    else:
        portfolio_df = pd.DataFrame(holdings, columns=["ticker", "weight"])
        portfolio_with_stats = portfolio_df.merge(ticker_summary, on="ticker", how="left")

        weighted_5d = (portfolio_with_stats["avg_return_5d"] * portfolio_with_stats["weight"] / 100).sum()
        weighted_30d = (portfolio_with_stats["avg_return_30d"] * portfolio_with_stats["weight"] / 100).sum()
        weighted_drawdown = (portfolio_with_stats["worst_drawdown"] * portfolio_with_stats["weight"] / 100).sum()

        exposure_scores = portfolio_with_stats["worst_drawdown"].abs() * portfolio_with_stats["weight"] / 100
        most_exposed = portfolio_with_stats.loc[exposure_scores.idxmax(), "ticker"]

        st.subheader("📊 Historical Risk Summary")
        st.caption("Based on S&P 500 performance during 45 geopolitical spike events (2016–2026)")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Change (5-day)", f"{weighted_5d:+.2f}%")
        with col2:
            st.metric("Avg Change (30-day)", f"{weighted_30d:+.2f}%")
        with col3:
            st.metric("Worst Drawdown", f"{weighted_drawdown:.2f}%")
        with col4:
            st.metric("Most Exposed", most_exposed)

        st.subheader("Holdings Breakdown")
        fig_holdings = px.bar(
            portfolio_with_stats,
            x="ticker",
            y=["avg_return_5d", "worst_drawdown"],
            barmode="group",
            title="5-Day Return vs Worst Drawdown by Holding",
            color_discrete_map={
                "avg_return_5d": "#2E75B6",
                "worst_drawdown": "#C00000"
            }
        )
        st.plotly_chart(fig_holdings, use_container_width=True)

        st.dataframe(
            portfolio_with_stats[["ticker", "weight", "avg_return_5d", "avg_return_30d", "worst_drawdown"]] \
                .rename(columns={
                    "avg_return_5d": "Avg Return 5d (%)",
                    "avg_return_30d": "Avg Return 30d (%)",
                    "worst_drawdown": "Worst Drawdown (%)"
                }),
            use_container_width=True
        )

# ── Section 4: Ticker Reaction Patterns ───────────────────────────
st.header("🗺️ Ticker Reaction Patterns")
st.caption("Average 5-day return during geopolitical spike events")

top_gainers = ticker_summary.nlargest(15, "avg_return_5d")[["ticker", "avg_return_5d", "worst_drawdown"]]
top_losers = ticker_summary.nsmallest(15, "avg_return_5d")[["ticker", "avg_return_5d", "worst_drawdown"]]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top Gainers (5-day)")
    fig_gainers = px.bar(
        top_gainers.sort_values("avg_return_5d"),
        x="avg_return_5d",
        y="ticker",
        orientation="h",
        color="avg_return_5d",
        color_continuous_scale="Blues",
        title="Top 15 Tickers by 5-Day Return"
    )
    st.plotly_chart(fig_gainers, use_container_width=True)

with col2:
    st.subheader("Top Losers (5-day)")
    fig_losers = px.bar(
        top_losers.sort_values("avg_return_5d", ascending=False),
        x="avg_return_5d",
        y="ticker",
        orientation="h",
        color="avg_return_5d",
        color_continuous_scale="Reds_r",
        title="Bottom 15 Tickers by 5-Day Return"
    )
    st.plotly_chart(fig_losers, use_container_width=True)

st.caption("Data updated: April 2026 | Source: GDELT, yfinance | NYU Big Data Project")