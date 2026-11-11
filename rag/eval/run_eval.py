"""Evaluate the Q&A system and print the REAL numbers.

Metrics
  1. Retrieval hit rate @k (k = 1, 3, 5): for answerable questions, did at
     least one expected document appear in the top k?
  2. Answer accuracy: does the answer contain the expected answer
     (sector / country / date / number from the data)?
  3. Groundedness, checked automatically:
       - every [doc_id] the model cites must be a retrieved document;
       - every number and date in the answer must appear in the retrieved
         documents (a number counts if a document number rounds to it).
     Answers with no numbers/dates can't be checked this way and are listed
     in results.md for manual review.
  4. Refusal accuracy: on unanswerable questions, did it refuse (or, for
     advice/prediction questions, decline advice)? Also reports false
     refusals on answerable questions.

Run:
    python -m rag.eval.run_eval --retrieval-only   # no API key needed
    python -m rag.eval.run_eval                    # full eval, needs GEMINI_API_KEY (default)
                                                   # or GEO_QA_PROVIDER=anthropic + ANTHROPIC_API_KEY
    python -m rag.eval.run_eval --retrieval-only --retriever dense   # baseline to compare
If a free-tier quota stops the run, re-run the same command later: answers
already received are cached in rag/eval/.answer_cache.json and skipped.
Writes rag/eval/results*.md (summary) and results*.json (every question);
the suffix shows the mode, e.g. results_dense_retrieval.md.
"""
import argparse
import datetime as dt
import json
import re
from pathlib import Path

from rag import config
from rag.generate import ADVICE_DECLINE, CITATION_RE, RateLimited, answer, call_llm
from rag.retrieve import Retriever

EVAL_DIR = Path(__file__).parent
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
# LLM answers already received, so a run stopped by a free-tier quota can
# continue later without paying for (or waiting on) the same calls again.
CACHE_PATH = EVAL_DIR / ".answer_cache.json"
KS = [1, 3, 5]

DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")


def ratio(num, den):
    return f"{num}/{den} ({100 * num / den:.1f}%)" if den else "n/a (0 questions)"


def numbers_in(text):
    """All numbers in text, dates removed first, as (value, decimals)."""
    text = DATE_RE.sub(" ", text)
    out = []
    for m in NUM_RE.finditer(text):
        s = m.group().replace(",", "")
        try:
            out.append((float(s), len(s.split(".")[1]) if "." in s else 0))
        except ValueError:
            pass
    return out


def check_grounding(answer_text, hits):
    """Return (facts_checked, unsupported_facts)."""
    body = CITATION_RE.sub(" ", answer_text)          # citations aren't claims
    docs_text = " ".join(h.doc["text"] for h in hits)
    doc_dates = set(DATE_RE.findall(docs_text))
    doc_nums = [v for v, _ in numbers_in(docs_text)]

    facts, unsupported = [], []
    for d in DATE_RE.findall(body):
        facts.append(d)
        if d not in doc_dates:
            unsupported.append(d)
    for value, decimals in numbers_in(body):
        facts.append(value)
        if not any(round(abs(x), decimals) == abs(value) for x in doc_nums):
            unsupported.append(value)
    return facts, unsupported


def contains_expected(text, expected):
    if expected is None:
        return False
    body = CITATION_RE.sub(" ", text).lower()
    if expected.lower() in body:
        return True
    # numbers: accept the same value written with fewer decimals or without "+"
    try:
        target = float(expected.replace("%", "").replace("+", ""))
    except ValueError:
        return False
    return any(round(target, d) == v or round(abs(target), d) == v
               for v, d in numbers_in(body))


def suggestion(name, value):
    tips = {
        "hit@5": "Add the event label and date to the start of every document, or add a "
                 "keyword (BM25) retriever and combine its scores with the embeddings.",
        "accuracy": "Look at the failed questions in results.md; if retrieval was right, "
                    "tighten the prompt; if not, fix retrieval first.",
        "grounded": "Lower max sentences in the prompt and ask the model to copy numbers "
                    "exactly as written in the documents.",
        "refusal": "Raise GEO_QA_MIN_SIM (see the similarity numbers above) so weak "
                   "matches are refused before the LLM is called.",
        "false_refusal": "Lower GEO_QA_MIN_SIM or raise k so the right document is retrieved.",
    }
    return f"  -> {name} is low ({value:.1f}%). Suggested fix: {tips[name]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true",
                    help="only measure retrieval (no LLM calls, no API key needed)")
    ap.add_argument("--k", type=int, default=config.TOP_K, help="documents given to the LLM")
    ap.add_argument("--retriever", choices=["hybrid", "dense"], default=config.RETRIEVER,
                    help="hybrid = embeddings + BM25 (default); dense = embeddings only")
    args = ap.parse_args()

    with open(QUESTIONS_PATH) as f:
        questions = [json.loads(line) for line in f if line.strip()]
    retriever = Retriever(mode=args.retriever)
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    # ── 1. Retrieval ──────────────────────────────────────────────
    hits_at = {k: 0 for k in KS}
    top1_ans, top1_unans = [], []
    records = {}
    for q in questions:
        hits = retriever.search(q["question"], k=max(KS + [args.k]))
        ids = [h.doc["doc_id"] for h in hits]
        best = max(h.score for h in hits[:args.k])  # what the refusal gate sees
        rec = {**q, "retrieved": ids[:args.k], "best_similarity": round(best, 4)}
        if q["answerable"]:
            top1_ans.append(best)
            for k in KS:
                if set(q["expected_doc_ids"]) & set(ids[:k]):
                    hits_at[k] += 1
            rec["hit@5"] = bool(set(q["expected_doc_ids"]) & set(ids[:5]))
        else:
            top1_unans.append(best)
        records[q["id"]] = rec

    lines = [f"# Geo-Risk Q&A eval — {dt.date.today()}", "",
             f"- Questions: {len(questions)} ({len(answerable)} answerable, {len(unanswerable)} unanswerable)",
             f"- Retriever: **{args.retriever}**; embedding model: `{config.EMBED_MODEL}`",
             f"- LLM: {config.LLM_PROVIDER} / `{config.LLM_MODEL}`"
             + (" (not used)" if args.retrieval_only else ""),
             f"- k given to LLM: {args.k}; refusal similarity gate: {config.MIN_SIMILARITY}", "",
             "| Metric | Result |", "|---|---|"]
    lows = []
    for k in KS:
        lines.append(f"| Retrieval hit rate @{k} | {ratio(hits_at[k], len(answerable))} |")
    if answerable and 100 * hits_at[5] / len(answerable) < 80:
        lows.append(suggestion("hit@5", 100 * hits_at[5] / len(answerable)))

    sim_note = (f"Best cosine similarity in top {args.k} — answerable: min {min(top1_ans):.3f}, mean "
                f"{sum(top1_ans) / len(top1_ans):.3f}; unanswerable: max {max(top1_unans):.3f}, "
                f"mean {sum(top1_unans) / len(top1_unans):.3f}") if top1_ans and top1_unans else ""

    # ── 2-4. Generation ───────────────────────────────────────────
    manual = []
    if not args.retrieval_only:
        cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}

        def cached_llm(question, hits):
            key = "|".join([config.LLM_PROVIDER, config.LLM_MODEL, args.retriever,
                            ",".join(h.doc["doc_id"] for h in hits), question])
            if key not in cache:
                cache[key] = call_llm(question, hits)
                CACHE_PATH.write_text(json.dumps(cache, indent=1))
            return cache[key]

        # First pass: make sure every answer exists (stop cleanly on quota).
        done = 0
        for q in questions:
            try:
                answer(q["question"], retriever=retriever, k=args.k, llm=cached_llm)
                done += 1
            except RateLimited as e:
                print(f"\nStopped: API quota reached after {done}/{len(questions)} questions "
                      f"(answers so far are saved).\n{e}\n"
                      "Re-run the same command later (free-tier daily limits reset at "
                      "midnight Pacific) and it will continue where it stopped.")
                raise SystemExit(1)
        print(f"All {len(questions)} answers ready ({config.LLM_PROVIDER} / {config.LLM_MODEL}).\n")

        correct = grounded = checkable = false_ref = refused_ok = invalid_cit = 0
        for q in questions:
            a = answer(q["question"], retriever=retriever, k=args.k, llm=cached_llm)
            rec = records[q["id"]]
            rec.update(answer=a.text, refused=a.refused, gated=a.gated,
                       cited=a.cited_ids, invalid_citations=a.invalid_citations)
            declined_advice = ADVICE_DECLINE.lower() in a.text.lower()
            if q["answerable"]:
                rec["correct"] = contains_expected(a.text, q["expected_answer"])
                correct += rec["correct"]
                if a.refused:
                    false_ref += 1
                    continue
                facts, unsupported = check_grounding(a.text, a.hits)
                ok = not unsupported and not a.invalid_citations and bool(a.cited_ids)
                invalid_cit += bool(a.invalid_citations)
                rec.update(facts_checked=[str(f) for f in facts],
                           unsupported=[str(u) for u in unsupported], grounded=ok)
                if facts:
                    checkable += 1
                    grounded += ok
                else:
                    manual.append(q["id"])
            else:
                ok = (a.refused and "refusal" in q["accept"]) or \
                     (declined_advice and "advice_decline" in q["accept"])
                rec["refusal_correct"] = bool(ok)
                refused_ok += ok

        n_ans, n_un = len(answerable), len(unanswerable)
        lines += [
            f"| Answer accuracy (answerable) | {ratio(correct, n_ans)} |",
            f"| Grounded answers (auto-checkable) | {ratio(grounded, checkable)} |",
            f"| Answers citing a non-retrieved doc | {ratio(invalid_cit, n_ans - false_ref)} |",
            f"| Refusal accuracy (unanswerable) | {ratio(refused_ok, n_un)} |",
            f"| False refusals (answerable) | {ratio(false_ref, n_ans)} |",
        ]
        for name, num, den, low_is_bad in [("accuracy", correct, n_ans, True),
                                           ("grounded", grounded, checkable, True),
                                           ("refusal", refused_ok, n_un, True),
                                           ("false_refusal", false_ref, n_ans, False)]:
            if not den:
                continue
            v = 100 * num / den
            if (low_is_bad and v < 80) or (not low_is_bad and v > 20):
                lows.append(suggestion(name, v))
    else:
        lines.append("| Generation metrics | not run (--retrieval-only) |")

    if sim_note:
        lines += ["", sim_note]
    if lows:
        lines += ["", "## Weak spots", *lows]
    lines += ["", "## Failed or unchecked questions"]
    for r in records.values():
        flags = []
        if r["answerable"] and not r.get("hit@5"):
            flags.append("expected doc not in top 5")
        if r.get("correct") is False:
            flags.append(f"expected '{r['expected_answer']}'")
        if r.get("unsupported"):
            flags.append(f"unsupported facts {r['unsupported']}")
        if r.get("refusal_correct") is False:
            flags.append("did not refuse")
        if r["id"] in manual:
            flags.append("no numbers/dates — review manually")
        if flags:
            lines.append(f"- **{r['id']}** {r['question']} — {'; '.join(flags)}")
            if r.get("answer"):
                lines.append(f"  - answer: {r['answer']}")

    report = "\n".join(lines)
    print(report)
    suffix = "" if args.retriever == "hybrid" else f"_{args.retriever}"
    if args.retrieval_only:
        suffix += "_retrieval"
    (EVAL_DIR / f"results{suffix}.md").write_text(report + "\n")
    with open(EVAL_DIR / f"results{suffix}.json", "w") as f:
        json.dump(list(records.values()), f, indent=2, ensure_ascii=False)
    print(f"\nSaved {EVAL_DIR / f'results{suffix}.md'} and results{suffix}.json")


if __name__ == "__main__":
    main()
