"""Offline indexing: load candidates → render probe texts → embed → write to Chroma."""
from __future__ import annotations

import time

from app.chroma_store import ChromaStore
from app.db import fetch_all_bundles
from app.embeddings import EmbeddingClient
from app.probe_texts import render_views


def build_index(
    dsn: str,
    api_key: str,
    embedding_model: str,
    store: ChromaStore,
    batch_size: int = 64,
) -> dict:
    """Build the Chroma index. Returns a summary dict."""
    t0 = time.monotonic()
    store.reset()

    bundles = fetch_all_bundles(dsn)
    embedder = EmbeddingClient(api_key=api_key, model=embedding_model)

    # Stage all (text, id, metadata) triples
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for b in bundles:
        cid = str(b["candidate"]["id"])
        views = render_views(b)
        for view_name, text in views.items():
            ids.append(f"{cid}::{view_name}")
            texts.append(text)
            metadatas.append({"candidate_id": cid, "view": view_name})

    # Embed in batches and upsert
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        embs = embedder.embed_batch(batch_texts)
        store.upsert_batch(
            ids=ids[i : i + batch_size],
            embeddings=embs,
            documents=batch_texts,
            metadatas=metadatas[i : i + batch_size],
        )

    return {
        "candidates_indexed": len(bundles),
        "documents_written":  len(texts),
        "duration_seconds":   round(time.monotonic() - t0, 2),
    }
