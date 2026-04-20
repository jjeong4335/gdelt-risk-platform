"""
Auto-generate spike event labels from GDELT DOC API article titles.
Runs once and saves labels to a JSON file for use in the dashboard.
"""

import os
import re
import json
import time
import requests
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta

GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

STOPWORDS = {
    "the","a","an","in","of","to","and","for","is","are","was","were",
    "on","at","by","with","as","it","be","this","that","from","or","but",
    "not","have","has","had","will","can","its","their","about","over",
    "after","before","into","than","more","also","been","when","who",
    "what","how","new","says","say","said","two","one","us","up","out",
    "after","during","amid","just","get","got","his","her","him","they",
    "them","we","our","your","my","he","she","would","could","should",
    "may","might","must","now","then","here","there","where","which",
    "were","per","ago","yet","via","due","war","news","world","day",
    "year","time","first","last","next","back","long","high","old",
    "gen","mr","dr","mrs","pm","am",
}

CATEGORY_QUERIES = {
    "POLITICAL_INSTABILITY": "political unrest government crisis protest",
    "CONFLICT":              "military conflict war attack violence",
    "SANCTIONS":             "economic sanctions trade restrictions",
    "PROTEST":               "protest demonstration civil unrest",
    "DIPLOMACY":             "diplomatic relations international agreement",
    "ECONOMIC_CRISIS":       "economic crisis financial market collapse",
}


def fetch_titles(query: str, date: str, window_days: int = 2,
                 max_records: int = 30, retry: int = 3) -> list[str]:
    """Fetch article titles from GDELT DOC API."""
    spike_dt = datetime.strptime(date, "%Y-%m-%d")
    start_dt = spike_dt - timedelta(days=window_days)
    end_dt   = spike_dt + timedelta(days=window_days)

    params = {
        "query":         f"{query} sourcelang:english",
        "mode":          "artlist",
        "maxrecords":    max_records,
        "startdatetime": start_dt.strftime("%Y%m%d000000"),
        "enddatetime":   end_dt.strftime("%Y%m%d235959"),
        "format":        "json",
    }

    for attempt in range(retry):
        try:
            r = requests.get(GDELT_API_URL, params=params, timeout=30)
            if r.status_code == 429:
                wait = 10 * (2 ** attempt)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            articles = r.json().get("articles", [])
            return [a["title"] for a in articles
                    if a.get("language", "").lower() == "english" and a.get("title")]
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(5)
    return []


def extract_label(titles: list[str], top_n: int = 3) -> str:
    """Extract top keywords from titles as event label."""
    words = []
    for title in titles:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", title.lower())
        words.extend([w for w in tokens if w not in STOPWORDS])

    top = Counter(words).most_common(top_n)
    if not top:
        return "Geopolitical Event"
    return " · ".join([w.title() for w, _ in top])


def main():
    spike_path = os.path.expanduser("~/dashboard_data/spike_events.parquet")
    output_path = os.path.expanduser("~/gdelt-risk-platform/rag/spike_labels.json")

    spikes = pd.read_parquet(spike_path)
    labels = {}

    total = len(spikes)
    for i, (_, row) in enumerate(spikes.iterrows()):
        date     = str(row["date"])[:10]
        category = row.get("dominant_category", "POLITICAL_INSTABILITY")
        query    = CATEGORY_QUERIES.get(category, "geopolitical crisis conflict")

        print(f"[{i+1}/{total}] {date} | {category}")
        titles = fetch_titles(query, date)

        if titles:
            label = extract_label(titles)
        else:
            # Fallback: use category name
            label = category.replace("_", " ").title()

        labels[date] = label
        print(f"  → {label}")

        # Polite delay
        time.sleep(6)

    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nDone! Saved {len(labels)} labels to {output_path}")


if __name__ == "__main__":
    main()
