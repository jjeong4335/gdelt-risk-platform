"""Streamlit UI for Geo-Risk Q&A, shared by two entry points:

    streamlit run dashboard/app.py   -> third tab of the full dashboard
    streamlit run rag/qa_app.py      -> Q&A on its own (works with rebuilt data,
                                        which can't power the other two tabs)
"""
import datetime as _dt

import streamlit as st

from dashboard.constants import SECTOR_MAP


@st.cache_resource
def load_retriever():
    from rag.retrieve import Retriever
    return Retriever()


def render_qa():
    st.subheader("💬 Geo-Risk Q&A")
    st.caption("Answers come only from the 31 detected spike events and their sector "
               "reactions. Historical data only — not investment advice.")

    try:
        retriever = load_retriever()
    except Exception as e:
        retriever = None
        st.warning(f"Q&A index not ready: {e}\n\nRun `python -m rag.build_corpus` "
                   "and `python -m rag.index` first.")

    if retriever is not None:
        question = st.text_input(
            "Ask about a spike event or sector reaction",
            placeholder="Which sector fell most 5 days after the Russia-Ukraine War spike?")
        with st.expander("Filters (optional)"):
            from rag import config as rag_config
            f1, f2, f3 = st.columns([1, 1, 2])
            use_dates = f1.checkbox("Filter by date")
            date_range = f2.date_input(
                "Event date range",
                value=(_dt.date(2016, 1, 1), _dt.date(2026, 12, 31)),
                disabled=not use_dates)
            sectors = f3.multiselect("Sectors", list(SECTOR_MAP.keys()))
            top_k = st.slider("Documents to retrieve (k)", 1, 10, rag_config.TOP_K)

        if question:
            from rag.generate import answer as rag_answer
            start = end = None
            if use_dates and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                start, end = (d.strftime("%Y-%m-%d") for d in date_range)
            try:
                with st.spinner("Retrieving documents and asking the LLM..."):
                    ans = rag_answer(question, retriever=retriever, k=top_k,
                                     start_date=start, end_date=end, sectors=sectors or None)
            except Exception as e:
                st.error(f"Error: {e}")
            else:
                st.markdown("#### Answer")
                st.markdown(ans.display_text)
                if ans.gated:
                    st.caption("No document was similar enough to the question, "
                               "so the LLM was not called.")
                if ans.invalid_citations:
                    st.warning("The model cited documents that were not retrieved: "
                               + ", ".join(ans.invalid_citations))

                if ans.cited_ids:
                    st.markdown("#### Citations")
                    by_id = {h.doc["doc_id"]: h.doc for h in ans.hits}
                    for cid in ans.cited_ids:
                        d = by_id.get(cid)
                        if d:
                            src = d["source_url"] or "GDELT spike record"
                            st.markdown(f"- `{cid}` — {d['event_date']} · {d['label']} · {src}")

                st.markdown("#### Retrieved documents")
                for h in ans.hits:
                    with st.expander(f"{h.score:.3f} · {h.doc['doc_id']} · {h.doc['label']}"):
                        st.write(h.doc["text"])
                        for u in h.doc.get("urls", []):
                            st.markdown(f"- {u}")
