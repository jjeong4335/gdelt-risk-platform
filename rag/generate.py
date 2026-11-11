"""Step 4 — Ask an LLM to answer using ONLY the retrieved documents.

Provider is chosen with GEO_QA_PROVIDER: "gemini" (default; Google AI Studio
free tier, key in GEMINI_API_KEY) or "anthropic" (Claude, key in
ANTHROPIC_API_KEY). The prompt and every safety rail below are identical for
both, so the eval can compare them fairly.

Safety rails, in order:
  1. Retrieval gate: if no document is similar enough to the question
     (best cosine similarity < MIN_SIMILARITY), refuse WITHOUT calling the
     LLM. Cheap, and the model can't make something up if it is never asked.
  2. Prompt rules: answer only from the documents, cite every claim,
     use the exact refusal sentence when the documents don't answer it,
     never give investment advice.
  3. Citation check: the model cites document IDs like [event:2022-02-25].
     We verify each ID was actually retrieved, then replace it with the
     event date and source URL for display.

Run a single question from the terminal:
    python -m rag.generate "Which sector fell most after the 2022 Russia-Ukraine spike?"
"""
import os
import re
import sys
import time
from dataclasses import dataclass, field

from rag import config
from rag.retrieve import Retriever

ADVICE_DECLINE = "I can't give investment advice."

SYSTEM_PROMPT = f"""You answer questions about geopolitical tension spikes detected in GDELT \
news data (2016-2026) and how groups of S&P 500 stocks moved afterwards.

Rules — follow all of them:
1. Use ONLY the facts in the <documents> provided. Do not use outside knowledge, \
even if you know it. Do not guess or estimate numbers that are not written in a document.
2. After every sentence that states a fact, cite the document(s) it came from using \
their id in square brackets, e.g. [event:2022-02-25] or [sector:2022-02-25:Airlines].
3. If the documents do not contain the answer, reply with exactly this sentence and nothing else:
{config.REFUSAL_TEXT}
4. Never give investment advice (no buy/sell/hold suggestions, no predictions of future prices). \
If the question asks for advice or a prediction, start with "{ADVICE_DECLINE}" and then, only if \
the documents are relevant, state what the historical data shows, with citations.
5. "Day +N" means N calendar days after the spike; say so if it matters. \
Headlines are derived from article URLs, not full article text.
6. Be brief: at most 5 sentences."""


@dataclass
class Answer:
    question: str
    text: str                       # raw model output (with [doc_id] citations)
    display_text: str               # citations replaced by "date · URL"
    hits: list                      # retrieved documents (list[Hit])
    cited_ids: list = field(default_factory=list)
    invalid_citations: list = field(default_factory=list)  # cited but NOT retrieved
    refused: bool = False
    gated: bool = False             # True = refused by the similarity gate (no LLM call)


def format_documents(hits):
    return "\n".join(
        f'<document id="{h.doc["doc_id"]}" event_date="{h.doc["event_date"]}">\n{h.doc["text"]}\n</document>'
        for h in hits)


def build_user_message(question, hits):
    return f"<documents>\n{format_documents(hits)}\n</documents>\n\nQuestion: {question}"


CITATION_RE = re.compile(r"\[((?:event|sector):[^\]]+)\]")


def render_citations(text, hits):
    """Replace [doc_id] with [date · URL] so the reader can verify the source."""
    by_id = {h.doc["doc_id"]: h.doc for h in hits}

    def repl(m):
        ids = [i.strip() for i in m.group(1).split(",")]
        parts = []
        for i in ids:
            d = by_id.get(i)
            if d is None:
                parts.append(f"{i} (not in retrieved documents)")
            else:
                src = d["source_url"] or "GDELT spike record"
                parts.append(f"{d['event_date']} · {src}")
        return "[" + "; ".join(parts) + "]"

    return CITATION_RE.sub(repl, text)


def cited_doc_ids(text):
    ids = []
    for m in CITATION_RE.finditer(text):
        ids += [i.strip() for i in m.group(1).split(",")]
    return list(dict.fromkeys(ids))


class RateLimited(RuntimeError):
    """The provider's quota is used up (e.g. Gemini free tier's daily limit)."""


_last_call = 0.0


def _pace():
    """Wait so calls are at least GEO_QA_MIN_INTERVAL seconds apart
    (free tiers allow only a few requests per minute)."""
    global _last_call
    wait = config.MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def call_gemini(question, hits):
    from google import genai
    from google.genai import errors, types

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError("Set the GEMINI_API_KEY environment variable first "
                           "(create a key at aistudio.google.com).")
    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        # Output budget includes the model's internal "thinking" tokens,
        # so leave room; low thinking keeps answers fast and cheap.
        max_output_tokens=2048,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )
    for attempt in range(3):
        _pace()
        try:
            resp = client.models.generate_content(
                model=config.LLM_MODEL, contents=build_user_message(question, hits), config=cfg)
            return (resp.text or "").strip()
        except errors.APIError as e:
            if getattr(e, "code", None) != 429:
                raise
            msg = str(e)
            if "PerDay" in msg or "per day" in msg.lower() or attempt == 2:
                raise RateLimited(f"Gemini quota exhausted: {msg[:300]}") from e
            time.sleep(30 * (attempt + 1))  # per-minute limit: wait and retry


def call_claude(question, hits):
    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable first.")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    kwargs = dict(
        model=config.LLM_MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(question, hits)}],
    )
    # Note: current SDK/models don't take a temperature setting, so answers can
    # vary slightly between runs; the eval reports what a single run produced.
    _pace()
    try:
        resp = client.messages.create(**kwargs)
    except anthropic.RateLimitError as e:
        raise RateLimited(str(e)[:300]) from e
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def call_llm(question, hits):
    if config.LLM_PROVIDER == "anthropic":
        return call_claude(question, hits)
    if config.LLM_PROVIDER == "gemini":
        return call_gemini(question, hits)
    raise ValueError(f"Unknown GEO_QA_PROVIDER: {config.LLM_PROVIDER!r} (use gemini or anthropic)")


def answer(question, retriever=None, k=config.TOP_K, start_date=None, end_date=None, sectors=None,
           llm=None):
    """llm: optional replacement for call_llm (the eval passes a caching one)."""
    retriever = retriever or Retriever()
    hits = retriever.search(question, k=k, start_date=start_date, end_date=end_date, sectors=sectors)

    # Gate on the best cosine similarity among the retrieved documents
    # (in hybrid mode the first hit is not necessarily the most similar).
    if not hits or max(h.score for h in hits) < config.MIN_SIMILARITY:
        return Answer(question, config.REFUSAL_TEXT, config.REFUSAL_TEXT, hits,
                      refused=True, gated=True)

    text = (llm or call_llm)(question, hits)
    cited = cited_doc_ids(text)
    retrieved = {h.doc["doc_id"] for h in hits}
    return Answer(
        question=question,
        text=text,
        display_text=render_citations(text, hits),
        hits=hits,
        cited_ids=cited,
        invalid_citations=[c for c in cited if c not in retrieved],
        refused=config.REFUSAL_TEXT in text,
    )


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which sector fell the most after the Russia-Ukraine War spike?"
    a = answer(q)
    print(f"[{config.LLM_PROVIDER} / {config.LLM_MODEL}]\n")
    print(a.display_text)
    print("\nRetrieved:")
    for h in a.hits:
        print(f"  {h.score:.3f}  {h.doc['doc_id']}")
