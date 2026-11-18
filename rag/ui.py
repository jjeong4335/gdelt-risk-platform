"""Streamlit UI for Geo-Risk Q&A, shared by two entry points:

    streamlit run dashboard/app.py   -> third tab of the full dashboard
    streamlit run rag/qa_app.py      -> Q&A on its own
"""
import datetime as _dt
import html
from urllib.parse import urlparse

import streamlit as st

from dashboard.constants import SECTOR_MAP

# (button label, question sent to the model)
EXAMPLES = [
    ("📉 Hardest-hit sector: Russia-Ukraine War",
     "Which sector fell the most 5 days after the Russia-Ukraine War spike?"),
    ("✈️ Airlines 30 days after Israel-Hamas War",
     "How did Airlines stocks move 30 days after the Israel-Hamas War spike?"),
    ("🌍 Most-mentioned country: Sudan Civil War",
     "Which country was mentioned most in news coverage of the Sudan Civil War spike?"),
    ("🚫 \"Should I buy defense stocks?\" (refusal test)",
     "Should I buy defense stocks before the next crisis?"),
]

_CSS = """
<style>
.qa-answer { background:#ffffff; border:1px solid #e5e7eb; border-left:4px solid #2563eb;
             border-radius:10px; padding:18px 20px; font-size:16px; line-height:1.6; color:#111827; }
.qa-answer.refused { border-left-color:#9ca3af; color:#4b5563; }
.qa-cite { display:inline-block; background:#eff6ff; color:#1d4ed8; border-radius:6px;
           font-size:12px; font-weight:600; padding:0 6px; margin:0 1px; vertical-align:2px; }
.qa-src { font-size:14px; padding:6px 0; border-bottom:1px solid #f3f4f6; color:#374151; }
.qa-src b { color:#1d4ed8; }
.qa-meta { font-size:12px; color:#6b7280; margin-top:6px; }
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
    """Replace [doc_id, ...] citations with small numbered badges [1][2]."""
    from rag.generate import CITATION_RE

    def repl(m):
        ids = [i.strip() for i in m.group(1).split(",")]
        return "".join(f"<span class='qa-cite'>{order.index(i) + 1}</span>"
                       for i in ids if i in order)
    return CITATION_RE.sub(repl, html.escape(text, quote=False))


def render_qa():
    from rag import config as rag_config

    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown("## 💬 Ask the data")
    st.caption("Answers use only the 31 detected spike events and how six S&P 500 sector groups "
               "moved afterwards. Every fact is cited. Historical data only — not investment advice.")

    try:
        retriever = load_retriever()
    except Exception as e:
        st.warning(f"The Q&A index isn't built yet ({e}). Run `python -m rag.build_corpus` "
                   "and `python -m rag.index`, then reload.")
        return

    st.markdown("**Try an example**")
    cols = st.columns(len(EXAMPLES))
    for col, (label, ex) in zip(cols, EXAMPLES):
        if col.button(label, width="stretch", help=ex):
            st.session_state["qa_question"] = ex

    question = st.text_input("Or ask your own question", key="qa_question",
                             placeholder="e.g. How did Energy stocks react after the Iran War spike?")

    with st.expander("Options"):
        f1, f2, f3 = st.columns([1, 2, 2])
        use_dates = f1.checkbox("Limit dates")
        date_range = f2.date_input("Event dates", value=(_dt.date(2016, 1, 1), _dt.date(2026, 12, 31)),
                                   disabled=not use_dates)
        sectors = f3.multiselect("Sectors", list(SECTOR_MAP.keys()))
        top_k = st.slider("Documents given to the model", 1, 10, rag_config.TOP_K)

    if not question:
        return

    from rag.generate import answer as rag_answer
    start = end = None
    if use_dates and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = (d.strftime("%Y-%m-%d") for d in date_range)
    try:
        with st.spinner("Searching the events and asking the model…"):
            ans = rag_answer(question, retriever=retriever, k=top_k,
                             start_date=start, end_date=end, sectors=sectors or None)
    except Exception as e:
        msg = str(e)
        if "busy" in msg or "quota" in msg.lower():
            st.warning("The model is busy or today's free quota is used up. Please try again in a "
                       "minute (or tomorrow if the daily quota ran out).")
        elif "API_KEY" in msg:
            st.warning("No API key found. Set GEMINI_API_KEY in the terminal, then restart the app.")
        else:
            st.error(f"Something went wrong: {msg[:300]}")
        return

    by_id = {h.doc["doc_id"]: h.doc for h in ans.hits}
    order = [c for c in ans.cited_ids if c in by_id]

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Answer")
        cls = "qa-answer refused" if ans.refused else "qa-answer"
        st.markdown(f"<div class='{cls}'>{_numbered(ans.text, order)}</div>", unsafe_allow_html=True)
        meta = f"Model: {rag_config.LLM_PROVIDER} / {rag_config.LLM_MODEL} · {len(ans.hits)} documents searched"
        if ans.gated:
            meta = "No event matched the question closely enough, so the model wasn't asked."
        st.markdown(f"<div class='qa-meta'>{meta}</div>", unsafe_allow_html=True)
        if ans.invalid_citations:
            st.warning("The model cited documents that weren't retrieved: "
                       + ", ".join(ans.invalid_citations))

    with right:
        st.markdown("#### Sources")
        if not order:
            st.caption("No sources cited.")
        for n, cid in enumerate(order, 1):
            d = by_id[cid]
            what = f"{d['sector']} after {d['label']}" if d.get("sector") else d["label"]
            link = (f"<a href='{html.escape(d['source_url'])}' target='_blank'>{_domain(d['source_url'])}</a>"
                    if d.get("source_url") else "GDELT spike record")
            st.markdown(f"<div class='qa-src'><b>[{n}]</b> {d['event_date']} · {html.escape(what)}"
                        f"<br><span class='qa-meta'>News: {link}</span></div>", unsafe_allow_html=True)

    with st.expander(f"See the {len(ans.hits)} documents the model was given"):
        for h in ans.hits:
            d = h.doc
            st.markdown(f"**{d['event_date']} · {d['label']}**"
                        + (f" · {d['sector']}" if d.get("sector") else "")
                        + f"  \n<span class='qa-meta'>relevance {h.score:.2f} · id `{d['doc_id']}`</span>",
                        unsafe_allow_html=True)
            st.write(d["text"])
            st.divider()
