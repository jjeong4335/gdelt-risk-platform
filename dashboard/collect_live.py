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

LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
GKG_URL = "http://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"


def latest_gkg_zip(max_back=4):
    """Return (zip bytes, timestamp) of the newest GKG file that actually exists.

    lastupdate.txt is sometimes updated before the file itself is uploaded, so
    the listed URL can return an error page instead of a zip. In that case step
    back 15 minutes at a time (up to max_back files)."""
    resp = requests.get(LASTUPDATE_URL, timeout=10)
    listed = [l.split()[-1] for l in resp.text.strip().split("\n") if "gkg.csv.zip" in l][0]
    ts = datetime.strptime(re.search(r"(\d{14})", listed).group(1), "%Y%m%d%H%M%S")
    last_err = None
    for i in range(max_back):
        t = ts - pd.Timedelta(minutes=15 * i)
        url = GKG_URL.format(ts=t.strftime("%Y%m%d%H%M%S"))
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and r.content[:2] == b"PK":  # zip files start with "PK"
                return r.content, t
            last_err = f"HTTP {r.status_code} for {url}"
        except requests.RequestException as e:
            last_err = str(e)
    raise RuntimeError(f"No GKG file available in the last {15 * max_back} min ({last_err})")


def fetch_latest_gdelt():
    try:
        content, _ = latest_gkg_zip()
        z = zipfile.ZipFile(io.BytesIO(content))
        fname = [f for f in z.namelist() if f.endswith(".csv")][0]
        return pd.read_csv(z.open(fname), sep="\t", header=None, on_bad_lines="skip")
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
