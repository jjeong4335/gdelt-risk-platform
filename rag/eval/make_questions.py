"""Generate the eval set FROM THE DATA (rag/data/corpus.jsonl).

Every expected answer is computed from the corpus, never typed by hand, so
the eval can't drift from the real numbers. Questions are picked
deterministically (same corpus -> same questions).

Each question records:
  expected_doc_ids  documents that contain the answer (for hit rate @k)
  expected_answer   short string/number that a correct answer must contain
  answerable        False for the questions the data cannot answer

Run:  python -m rag.eval.make_questions
Writes rag/eval/questions.jsonl (commit it; you can also edit it by hand).
"""
import json
from collections import Counter
from pathlib import Path

from rag import config
from rag.index import load_corpus

OUT_PATH = Path(__file__).parent / "questions.jsonl"

# Questions the data cannot answer. `accept` = which replies count as a
# correct refusal.
UNANSWERABLE = [
    {"question": "Should I buy defense stocks before the next geopolitical crisis?",
     "category": "advice", "accept": ["refusal", "advice_decline"]},
    {"question": "Which sector will fall the most after the next Iran escalation?",
     "category": "prediction", "accept": ["refusal", "advice_decline"]},
    {"question": "How did Bitcoin react to the Russia-Ukraine War spike?",
     "category": "not_in_data", "accept": ["refusal"]},
    {"question": "What was the Geo-Tension Index during the 2008 financial crisis?",
     "category": "out_of_range", "accept": ["refusal"]},
    {"question": "What interest rate decision did the Federal Reserve make in March 2023?",
     "category": "not_in_data", "accept": ["refusal"]},
]


def spread(items, n):
    """Pick n items evenly spaced through the list (deterministic)."""
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def pct(x):
    return f"{x:+.2f}%"


def make(docs):
    events = sorted((d for d in docs if d["type"] == "event"), key=lambda d: d["event_date"])
    sector_docs = {(d["event_date"], d["sector"]): d for d in docs if d["type"] == "event_sector"}
    sectors_of = {}
    for (date, sector) in sector_docs:
        sectors_of.setdefault(date, []).append(sector)
    with_sectors = [e for e in events if e["event_date"] in sectors_of]
    label_counts = Counter(e["label"] for e in events)

    def name(e):
        """Use the label alone only when it identifies exactly one event."""
        if not e["label"].startswith("Unlabeled") and label_counts[e["label"]] == 1:
            return f"the {e['label']} spike"
        return f"the spike on {e['event_date']}"

    qs = []

    # A. Weakest sector at Day +5 (6 questions)
    for e in spread(with_sectors, 6):
        date = e["event_date"]
        day5 = [(s, sector_docs[(date, s)]["returns"].get("5")) for s in sectors_of[date]]
        day5 = [(s, c["return_pct"]) for s, c in day5 if c]
        if not day5:
            continue
        weakest = min(day5, key=lambda x: x[1])[0]
        qs.append({"question": f"Which sector fell the most 5 days after {name(e)}?",
                   "category": "weakest_sector_d5", "expected_answer": weakest,
                   "expected_doc_ids": [e["doc_id"], f"sector:{date}:{weakest}"]})

    # B. One sector at Day +30 (6 questions, cycling through sectors)
    all_sectors = sorted({s for _, s in sector_docs})
    for i, e in enumerate(spread(with_sectors[1:], 6)):
        date = e["event_date"]
        avail = sorted(sectors_of[date])
        sector = all_sectors[i % len(all_sectors)]
        sector = sector if sector in avail else avail[0]
        cell = sector_docs[(date, sector)]["returns"].get("30")
        if not cell:
            continue
        qs.append({"question": f"How did {sector} stocks move 30 days after {name(e)}?",
                   "category": "sector_d30", "expected_answer": pct(cell["return_pct"]),
                   "expected_doc_ids": [f"sector:{date}:{sector}"]})

    has_index = all(e.get("geo_tension_index") is not None for e in events)
    has_z = all(e.get("z_score") is not None for e in events)

    # C. Index level on a date (4 questions; original data only)
    for e in (spread(events[2:], 4) if has_index else []):
        qs.append({"question": f"What was the Geo-Tension Index on {e['event_date']}?",
                   "category": "index_value", "expected_answer": f"{e['geo_tension_index']:.2f}",
                   "expected_doc_ids": [e["doc_id"]]})

    # D. Most distinctive country (3 questions). Asked by date, not by event
    # name: names are built from these countries, so naming the event would
    # give the answer away.
    for e in spread([e for e in events if e["countries"]][3:], 3):
        qs.append({"question": f"Which country stood out most in news coverage of the spike on {e['event_date']}?",
                   "category": "top_country", "expected_answer": e["countries"][0],
                   "expected_doc_ids": [e["doc_id"]]})

    # E. Label-only phrasing, Day +1 (3 questions) — tests retrieval without a date
    unique = [e for e in with_sectors
              if not e["label"].startswith("Unlabeled") and label_counts[e["label"]] == 1]
    for e in spread(unique, 3):
        date = e["event_date"]
        sector = "Defense" if "Defense" in sectors_of[date] else sorted(sectors_of[date])[0]
        cell = sector_docs[(date, sector)]["returns"].get("1")
        if not cell:
            continue
        qs.append({"question": f"How did {sector} stocks react one day after the {e['label']} spike?",
                   "category": "label_only_d1", "expected_answer": pct(cell["return_pct"]),
                   "expected_doc_ids": [f"sector:{date}:{sector}"]})

    # C'. Without index values: strongest sector at Day +5 (4 questions)
    for e in ([] if has_index else spread(with_sectors[2:], 4)):
        date = e["event_date"]
        day5 = [(s, sector_docs[(date, s)]["returns"].get("5")) for s in sectors_of[date]]
        day5 = [(s, c["return_pct"]) for s, c in day5 if c]
        if not day5:
            continue
        best = max(day5, key=lambda x: x[1])[0]
        qs.append({"question": f"Which sector rose the most 5 days after {name(e)}?",
                   "category": "strongest_sector_d5", "expected_answer": best,
                   "expected_doc_ids": [e["doc_id"], f"sector:{date}:{best}"]})

    # F. Questions that need comparing many events — hard for top-k RAG.
    # Only asked when the answer is unique: recomputed index values are
    # capped at 10.00, so several spikes can tie for "highest", and a question
    # with several right answers would be scored unfairly.
    def add_unique_max(question, items, value, summary_ids=()):
        summary_ids = list(summary_ids)
        items = [x for x in items if value(x) is not None]
        if not items:
            return False
        best = max(value(x) for x in items)
        winners = [x for x in items if value(x) == best]
        if len(winners) != 1:
            print(f"  skipped (tie for the top among {len(winners)}): {question}")
            return False
        qs.append({"question": question, "category": "aggregate",
                   "expected_answer": winners[0]["date"],
                   "expected_doc_ids": [winners[0]["doc"]] + summary_ids})
        return True

    ev_items = [{"date": e["event_date"], "doc": e["doc_id"],
                 "index": e.get("geo_tension_index"), "z": e.get("z_score")} for e in events]
    years = Counter(e["event_date"][:4] for e in events)
    year = max(years, key=lambda y: (years[y], y))
    in_year = [x for x in ev_items if x["date"].startswith(year)]
    added = 0
    if has_index:
        added += add_unique_max("Which detected spike had the highest Geo-Tension Index?",
                                ev_items, lambda x: x["index"])
        if has_z:
            added += add_unique_max(f"Which {year} spike had the largest z-score?",
                                    in_year, lambda x: x["z"], ["summary:events"])
        else:
            added += add_unique_max(f"Which {year} spike had the highest Geo-Tension Index?",
                                    in_year, lambda x: x["index"])
    if added < 2:
        airlines = [{"date": d, "doc": f"sector:{d}:Airlines", "ret": doc["returns"]["5"]["return_pct"]}
                    for (d, sct), doc in sector_docs.items() if sct == "Airlines" and "5" in doc["returns"]]
        add_unique_max("After which spike did Airlines stocks fall the most by Day +5?",
                       airlines, lambda x: -x["ret"], ["summary:Airlines"])

    for q in qs:
        q["answerable"] = True
    return _finish(qs)


def _finish(qs):
    for q in UNANSWERABLE:
        qs.append({**q, "answerable": False, "expected_answer": None, "expected_doc_ids": []})
    for i, q in enumerate(qs, 1):
        q["id"] = f"q{i:02d}"
    return qs


def main():
    qs = make(load_corpus(config.CORPUS_PATH))
    with open(OUT_PATH, "w") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    n_ans = sum(q["answerable"] for q in qs)
    print(f"Wrote {len(qs)} questions ({n_ans} answerable, {len(qs) - n_ans} unanswerable) to {OUT_PATH}")


if __name__ == "__main__":
    main()
