"""
Collect live GDELT data every 15 minutes and compute Geo-Tension score.
Saves to SQLite for dashboard display.
"""
import requests, zipfile, io, sqlite3, time, re
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

DB_PATH = "/home/jj4335_nyu_edu/dashboard_data/live_tension.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_tension (
            timestamp TEXT PRIMARY KEY,
            tension_score REAL
        )
    """)
    conn.commit()
    conn.close()

def fetch_latest_gdelt():
    try:
        resp = requests.get("http://data.gdeltproject.org/gdeltv2/lastupdate.txt", timeout=10)
        for line in resp.text.strip().split("\n"):
            if "gkg.csv.zip" in line:
                url = line.split()[-1]
                r = requests.get(url, timeout=30)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                fname = [f for f in z.namelist() if f.endswith(".csv")][0]
                df = pd.read_csv(z.open(fname), sep="\t", header=None, on_bad_lines="skip")
                return df
    except Exception as e:
        print(f"Fetch error: {e}")
    return None

def compute_tension(df):
    try:
        # Col 6: themes, Col 15: tone
        df = df[[6, 15]].dropna()
        df.columns = ["themes", "tone"]
        # International filter
        df = df[df["themes"].str.contains(r"#(?!US)[A-Z]{2}#", na=False, regex=True)]
        # Extract negative tone (3rd value)
        df["neg"] = df["tone"].str.split(",").str[2].astype(float)
        df = df[(df["neg"] >= 0) & (df["neg"] <= 50)]
        if len(df) == 0:
            return None
        avg_neg = df["neg"].mean()
        log_vol = np.log(len(df) + 1)
        raw = avg_neg * log_vol
        # Normalize using actual p1-p99 from raw geo_tension
        # Empirical raw values from historical data
        p1_raw = 10.5
        p99_raw = 35.0
        score = max(0, min(10, (raw - p1_raw) / (p99_raw - p1_raw) * 10))
        return round(score, 2)
    except Exception as e:
        print(f"Compute error: {e}")
        return None

def save_score(score):
    conn = sqlite3.connect(DB_PATH)
    et = pytz.timezone("America/New_York")
    ts = datetime.now(et).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("INSERT OR REPLACE INTO live_tension (timestamp, tension_score) VALUES (?, ?)", (ts, score))
    conn.commit()
    conn.close()
    print(f"{ts} → {score}")

if __name__ == "__main__":
    init_db()
    while True:
        df = fetch_latest_gdelt()
        if df is not None:
            score = compute_tension(df)
            if score is not None:
                save_score(score)
        time.sleep(900)  # 15 minutes
