"""Run the Geo-Risk Q&A on its own:  streamlit run rag/qa_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Geo-Risk Q&A", page_icon="💬", layout="wide")

from rag.ui import render_qa  # noqa: E402

render_qa()
