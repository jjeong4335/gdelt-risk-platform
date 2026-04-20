"""
Collect real-time GDELT tension scores every 15 minutes and store in SQLite.
Run as background process: nohup python3 collect_live.py &
"""

import sqlite3
import requests
import zipfile
import io
import time
import pandas as pd
from datetime import datetime

DB_PATH = "/home/jj4335_nyu_edu/dashboard_data/live_tension.db"
KEYWORDS = "MILITARY|WAR|WEAPON|MISSILE|SANCTION|EMBARGO|COUP|PROTEST|DIPLOMATIC|NUCLEAR"

def init_db():
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_tension (
            timestamp TEXT PRIMARY KEY,
            tension_score REAL,
            num_articles INTEGER,
            dominant_category TEXT
        )
    """)
    conn.commit()
    conn.close()

def fetch_tension():
    """Fetch latest GDELT data and compute tension score."""
    try:
        resp = requests.get("http://data.gdeltproject.org/gdeltv2/lastupdate.txt", timeout=10)
        lines = resp.text.strip().split("\n")
        gkg_url = [l.split()[-1] for l in lines if "gkg.csv.zip" in l][0]
        r = requests.get(gkg_url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        fname = z.namelist()[0]
        with z.open(fname) as f:
            df = pd.read_csv(f, sep="\t", header=None, on_bad_lines="skip",
                           usecols=[1, 7], names=["date", "themes"])
        # Filter geopolitical keywords
        df = df[df["themes"].str.contains(KEYWORDS, na=False, case=False)]
        num_articles = len(df)

        # Compute tension score (normalized)
        tension_score = round(num_articles / 50.0, 3)

        # Dominant category
        cats = {
            "Military":   ["MILITARY", "WAR", "WEAPON", "MISSILE"],
            "Sanctions":  ["SANCTION", "EMBARGO"],
            "Diplomatic": ["DIPLOMATIC"],
            "Nuclear":    ["NUCLEAR"],
            "Political":  ["COUP", "PROTEST"],
        }
        cat_counts = {}
        for cat, keywords in cats.items():
            cat_counts[cat] = df["themes"].str.contains("|".join(keywords), na=False).sum()
        dominant = max(cat_counts, key=cat_counts.get)

        return tension_score, num_articles, dominant

    except Exception as e:
        print(f"Error fetching GDELT: {e}")
        return None, None, None

def save_to_db(tension_score, num_articles, dominant_category):
    """Save tension score to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT OR REPLACE INTO live_tension
        (timestamp, tension_score, num_articles, dominant_category)
        VALUES (?, ?, ?, ?)
    """, (timestamp, tension_score, num_articles, dominant_category))
    conn.commit()
    conn.close()
    print(f"[{timestamp}] Saved: tension={tension_score}, articles={num_articles}, cat={dominant_category}")

def main():
    init_db()
    print("Live tension collector started. Collecting every 15 minutes...")
    while True:
        tension, articles, category = fetch_tension()
        if tension is not None:
            save_to_db(tension, articles, category)
        time.sleep(900)  # 15 minutes

if __name__ == "__main__":
    main()
