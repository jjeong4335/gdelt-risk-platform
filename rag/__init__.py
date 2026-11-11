"""Geo-Risk Q&A: a small Retrieval-Augmented Generation (RAG) layer on top of
the existing GDELT spike + sector-reaction outputs.

Pipeline (each step is its own module so it can be explained and tested alone):
    build_corpus.py  -> turn existing Parquet outputs into short text documents
    index.py         -> embed the documents and store them in a FAISS index
    retrieve.py      -> find the documents most similar to a question
    generate.py      -> ask Claude to answer ONLY from those documents
    eval/            -> measure retrieval, groundedness and refusals
"""
