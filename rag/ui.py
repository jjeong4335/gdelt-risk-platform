"""Geo-Risk Q&A as a chat panel.

    streamlit run dashboard/app.py   -> chat panel on the right of the dashboard
    streamlit run rag/qa_app.py      -> the same chat on its own page

Each question is answered on its own (retrieval + grounded answer); earlier
messages are shown for reference but are not sent to the model, so every
answer can still be checked against its own sources.
"""
import html
from urllib.parse import urlparse

import streamlit as st



EXAMPLES = [
    "What were the best-performing stocks when the tension index went up?",
    "Did the VIX or our Tension Index move first?",
    "Were the most volatile stocks similar across geopolitical events?",
]


def examples_from(docs):
    """Example questions shown in the chat window (label, question)."""
    return [(q, q) for q in EXAMPLES]

_CSS = """
<style>
.qa-title { font-size: 17px; font-weight: 800; color: #0f172a; }
.qa-intro { font-size: 12.5px; color: #64748b; line-height: 1.5; margin: 4px 0 12px; }
.qa-hint { font-size: 12px; font-weight: 600; color: #94a3b8; text-transform: uppercase;
           letter-spacing: 0.5px; margin: 6px 0 6px; }
.st-key-qa_examples button { justify-content: flex-start; text-align: left; border-radius: 12px;
                             padding: 8px 12px; background: #f8fafc; border: 1px solid #e2e8f0; }
.st-key-qa_examples button:hover { background: #e8f0fe; border-color: #93c5fd; }
.st-key-qa_examples button div, .st-key-qa_examples button p { justify-content: flex-start;
                             text-align: left; font-size: 13px; width: 100%; }
.qa-answer { font-size: 15px; line-height: 1.55; color: #111827; }
.qa-answer.refused { color: #6b7280; font-style: italic; }
.qa-cite { display:inline-block; background:#e8f0fe; color:#1d4ed8; border-radius:6px;
           font-size:11px; font-weight:600; padding:0 5px; margin:0 1px; vertical-align:2px; }
.qa-src { font-size: 12px; color:#4b5563; margin-top: 6px; line-height: 1.5; }
.qa-src b { color:#1d4ed8; }
.qa-meta { font-size: 11px; color:#9ca3af; margin-top: 4px; }
</style>
"""


@st.cache_resource
def load_retriever():
    from rag.retrieve import Retriever
    return Retriever()


def _domain(url):
    try:
        return urlparse(url).netloc.replace("www.", "") or url
    except Exception:
        return url


def _numbered(text, order):
    """Replace [doc_id, ...] citations with small numbered badges."""
    from rag.generate import CITATION_RE

    def repl(m):
        ids = [i.strip() for i in m.group(1).split(",")]
        return "".join(f"<span class='qa-cite'>{order.index(i) + 1}</span>" for i in ids if i in order)
    return CITATION_RE.sub(repl, html.escape(text, quote=False))


def _ask(question, retriever, k, start, end, sectors):
    """Run one question and return a plain dict that can live in session_state."""
    from rag import config
    from rag.generate import answer

    try:
        ans = answer(question, retriever=retriever, k=k, start_date=start, end_date=end,
                     sectors=sectors or None)
    except Exception as e:
        msg = str(e)
        if "busy" in msg or "quota" in msg.lower():
            text = ("The model is busy or today's free quota is used up. "
                    "Please try again in a minute (or tomorrow if the daily quota ran out).")
        elif "API_KEY" in msg:
            text = "No API key found. Set GEMINI_API_KEY in the terminal, then restart the app."
        else:
            text = f"Something went wrong: {msg[:300]}"
        return {"role": "assistant", "error": text}

    by_id = {h.doc["doc_id"]: h.doc for h in ans.hits}
    order = [c for c in ans.cited_ids if c in by_id]
    return {
        "role": "assistant",
        "html": _numbered(ans.text, order),
        "refused": ans.refused,
        "gated": ans.gated,
        "invalid": ans.invalid_citations,
        "model": f"{config.LLM_PROVIDER} / {config.LLM_MODEL}",
        "sources": [{"n": n, "date": by_id[c]["event_date"], "label": by_id[c]["label"],
                     "sector": None if by_id[c]["doc_id"].startswith("summary:") else by_id[c].get("sector"),
                     "url": by_id[c].get("source_url") or ""}
                    for n, c in enumerate(order, 1)],
        "docs": [{"date": h.doc["event_date"], "label": h.doc["label"], "sector": h.doc.get("sector"),
                  "score": h.score, "id": h.doc["doc_id"], "text": h.doc["text"]} for h in ans.hits],
    }


def _show(msg):
    if msg["role"] == "user":
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(msg["text"])
        return
    with st.chat_message("assistant", avatar=":material/insights:"):
        if "error" in msg:
            st.warning(msg["error"])
            return
        cls = "qa-answer refused" if msg["refused"] else "qa-answer"
        st.markdown(f"<div class='{cls}'>{msg['html']}</div>", unsafe_allow_html=True)
        if msg["sources"]:
            lines = []
            for s in msg["sources"]:
                what = f"{s['sector']} after {s['label']}" if s["sector"] else s["label"]
                link = (f" · <a href='{html.escape(s['url'])}' target='_blank'>{_domain(s['url'])}</a>"
                        if s["url"] else "")
                lines.append(f"<b>[{s['n']}]</b> {s['date']} · {html.escape(what)}{link}")
            st.markdown("<div class='qa-src'>" + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
        meta = ("No event matched closely enough, so the model wasn't asked."
                if msg["gated"] else f"{msg['model']} · {len(msg['docs'])} documents searched")
        st.markdown(f"<div class='qa-meta'>{meta}</div>", unsafe_allow_html=True)
        if msg["invalid"]:
            st.warning("Cited documents that weren't retrieved: " + ", ".join(msg["invalid"]))
        if msg["docs"]:
            with st.expander("Documents the model was given"):
                for d in msg["docs"]:
                    st.markdown(f"**{d['date']} · {d['label']}" + (f" · {d['sector']}" if d["sector"] else "")
                                + f"**  \n<span class='qa-meta'>relevance {d['score']:.2f} · `{d['id']}`</span>",
                                unsafe_allow_html=True)
                    st.caption(d["text"])


def render_chat(panel=True):
    """panel=True: a card that sits in the dashboard's right-hand column."""
    from rag import config

    history = st.session_state.setdefault("qa_chat", [])
    area = st.container(key="chat_panel") if panel else st.container()

    with area:
        st.markdown(_CSS, unsafe_allow_html=True)
        head, clear, close = st.columns([4, 1.2, 0.6], vertical_alignment="center")
        head.markdown("<div class='qa-title'>Ask the data</div>", unsafe_allow_html=True)
        if history and clear.button("Clear", type="tertiary", help="Clear the conversation"):
            st.session_state["qa_chat"] = []
            st.rerun()
        if panel and close.button("", icon=":material/close:", type="tertiary", key="chat_close",
                                  help="Close the assistant"):
            st.session_state.chat_open = False
            st.rerun()
        st.markdown("<div class='qa-intro'>Answers come only from the detected spike events and how "
                    "S&P 500 sectors moved afterwards, with sources for every fact. Historical "
                    "data only, not investment advice.</div>", unsafe_allow_html=True)

        try:
            retriever = load_retriever()
        except Exception as e:
            st.warning(f"The Q&A index isn't built yet ({e}). Run `python -m rag.build_corpus` "
                       "and `python -m rag.index`, then reload.")
            return

        if not history:
            st.markdown("<div class='qa-hint'>Try asking</div>", unsafe_allow_html=True)
            with st.container(key="qa_examples"):
                for label, q in examples_from(retriever.docs):
                    if st.button(label, key=f"ex_{label}", width="stretch", help=q):
                        st.session_state["qa_pending"] = q

        for msg in history:
            _show(msg)

        typed = st.chat_input("Ask about a spike event or sector…")
        question = typed or st.session_state.pop("qa_pending", None)
        if question:
            history.append({"role": "user", "text": question})
            _show(history[-1])
            with st.spinner("Searching the events and asking the model…"):
                history.append(_ask(question, retriever, config.TOP_K, None, None, None))
            st.rerun()


# Kept for older imports.
render_qa = render_chat
