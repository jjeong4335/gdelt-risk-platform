"""Step 4 — Ask Claude to answer using ONLY the retrieved documents.

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
    try:
        # temperature=0 makes answers as repeatable as possible for the eval.
        resp = client.messages.create(temperature=0, **kwargs)
    except anthropic.BadRequestError as e:
        if "temperature" not in str(e):
            raise
        resp = client.messages.create(**kwargs)  # model doesn't accept temperature
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def answer(question, retriever=None, k=config.TOP_K, start_date=None, end_date=None, sectors=None):
    retriever = retriever or Retriever()
    hits = retriever.search(question, k=k, start_date=start_date, end_date=end_date, sectors=sectors)

    if not hits or hits[0].score < config.MIN_SIMILARITY:
        return Answer(question, config.REFUSAL_TEXT, config.REFUSAL_TEXT, hits,
                      refused=True, gated=True)

    text = call_claude(question, hits)
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
    print(a.display_text)
    print("\nRetrieved:")
    for h in a.hits:
        print(f"  {h.score:.3f}  {h.doc['doc_id']}")
