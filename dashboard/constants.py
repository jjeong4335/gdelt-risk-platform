"""Shared constants used by both the Streamlit dashboard and the RAG Q&A code.

Kept in one place so the Q&A feature names events and sectors exactly the
same way the dashboard does.
"""

# ── Sector mapping ────────────────────────────────────────────────
SECTOR_MAP = {
    "Defense":         ["LMT", "RTX", "NOC", "GD", "HII", "BA", "TDG", "LHX"],
    "Energy":          ["XOM", "CVX", "COP", "OXY", "SLB", "EOG", "PSX", "VLO", "MPC"],
    "Gold/Safe Haven": ["NEM", "FCX", "KO", "PG", "JNJ", "CL", "GIS"],
    "Tech":            ["AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN", "NVDA", "ORCL", "CRM"],
    "Semiconductors":  ["AMD", "INTC", "QCOM", "NXPI", "AMAT", "LRCX", "MCHP", "TXN"],
    "Airlines":        ["AAL", "UAL", "DAL", "LUV", "NCLH", "RCL", "CCL"],
}


# ── Spike event labels ────────────────────────────────────────────
SPIKE_LABELS = {
    # 2018 - North Korea, Syria, Gaza
    "2018-01-02": "Iran Protests",
    "2018-04-16": "Syria Airstrikes",
    "2018-05-15": "Gaza Border Protests",
    # 2019 - Iran, Hong Kong
    "2019-07-01": "Iran Strait of Hormuz",
    "2019-08-13": "Hong Kong Protests",
    "2019-10-03": "Hong Kong Protests Peak",
    "2019-10-04": "Hong Kong Protests Peak",

    # 2022 - Russia-Ukraine War
    "2022-01-06": "Russia-Ukraine Tensions",
    "2022-02-13": "Russia-Ukraine Escalation",
    "2022-02-18": "Russia-Ukraine Escalation",
    "2022-02-25": "Russia-Ukraine War",
    "2022-07-09": "Sri Lanka Political Crisis",
    # 2023 - Sudan, Israel-Hamas
    "2023-03-28": "Russia-Ukraine · NATO · Israel",
    "2023-05-04": "Sudan Civil War",
    "2023-10-18": "Israel-Hamas War",
    "2023-11-24": "Israel-Hamas Ceasefire",
    # 2024 - Iran-Israel Escalation
    "2024-04-26": "Iran-Israel Tensions",
    "2024-04-29": "Iran-Israel Tensions",
    "2024-04-30": "Iran-Israel Tensions",
    "2024-05-01": "Iran-Israel Tensions",
    "2024-05-02": "Iran-Israel Tensions",
    "2024-08-01": "Russia Ukraine Escalation",
    "2024-08-05": "Russia Ukraine Escalation",
    # 2025 - Iran War, Geopolitical Tensions
    "2025-06-09": "Iran War",
    "2025-06-10": "Iran War",
    "2025-06-11": "Iran War",
    "2025-06-12": "Iran War",
    "2025-09-09": "Russia Ukraine Tensions",
    "2025-10-02": "Russia Ukraine Tensions",
    # 2026 - Iran War Escalation
    "2026-01-12": "Iran War Escalation",
    "2026-01-14": "Iran War Escalation",
}
