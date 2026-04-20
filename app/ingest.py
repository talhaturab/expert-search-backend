"""Offline indexing: load candidates → render probe texts → embed → write to Chroma.

Resumable, parallel, and rate-limit-safe:
- Embeds run across N worker threads for throughput (I/O-bound, OpenRouter limits
  are the ceiling).
- On start, reads existing Chroma IDs and skips any doc already embedded.
  You can kill the process and re-run; it picks up where it left off.
- If we hit a RateLimitError (or any OpenAI error the SDK can't auto-retry past),
  we cancel pending work and stop cleanly, returning what was done and a
  `stopped_reason` string. Re-run to continue.
- `reset=True` wipes the Chroma collection first (for the explicit "start fresh" case).
- Progress printed live to stdout (rate + ETA) so you can watch from the terminal.
"""
from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import APIError, RateLimitError

from app.chroma_store import ChromaStore
from app.db import fetch_all_bundles
from app.embeddings import EmbeddingClient
from app.probe_texts import render_views


log = logging.getLogger(__name__)

# How many concurrent embed-batch HTTP calls. OpenRouter handles 8 comfortably;
# bump if you see the server-side latency dominating and you have quota headroom.
DEFAULT_WORKERS = 8


def _existing_ids(store: ChromaStore) -> set[str]:
    """Return the set of doc IDs already in Chroma (ids only — no embeddings)."""
    try:
        data = store._collection.get(include=[])  # Chroma: ids are always returned
        return set(data.get("ids", []))
    except Exception:
        return set()


def _print_progress(processed: int, total: int, start_time: float) -> None:
    elapsed = time.monotonic() - start_time
    rate = processed / max(elapsed, 1e-6)
    remaining = (total - processed) / max(rate, 1e-6)
    bar_width = 30
    filled = int(bar_width * processed / max(total, 1))
    bar = "█" * filled + "·" * (bar_width - filled)
    sys.stdout.write(
        f"\r  [{bar}] {processed:5d}/{total} "
        f"({100.0 * processed / max(total, 1):5.1f}%)  "
        f"{rate:5.1f} docs/s  ETA {int(remaining):4d}s  elapsed {int(elapsed):4d}s"
    )
    sys.stdout.flush()


def _embed_and_upsert_batch(
    embedder: EmbeddingClient,
    store: ChromaStore,
    ids: list[str],
    texts: list[str],
    metas: list[dict],
) -> int:
    embs = embedder.embed_batch(texts)
    store.upsert_batch(ids=ids, embeddings=embs, documents=texts, metadatas=metas)
    return len(texts)


def build_index(
    dsn: str,
    api_key: str,
    embedding_model: str,
    store: ChromaStore,
    batch_size: int = 64,
    limit: int | None = None,
    reset: bool = False,
    max_workers: int = DEFAULT_WORKERS,
) -> dict:
    """Build (or resume building) the Chroma index.

    Args:
        limit: Optional cap on candidates to ingest. None = ingest all 10k.
        reset: If True, wipe the Chroma collection first (fresh start).
               If False, skip any doc whose ID is already in Chroma (resume).
        max_workers: Parallel embed-batch HTTP calls.

    Returns a dict with:
        candidates_loaded, documents_to_process, documents_written,
        documents_skipped_existing, stopped_reason, duration_seconds.
    """
    t0 = time.monotonic()
    if reset:
        log.info("Resetting Chroma collection (fresh start)...")
        store.reset()

    bundles = fetch_all_bundles(dsn, limit=limit)
    embedder = EmbeddingClient(api_key=api_key, model=embedding_model)

    # Build the full list of (id, text, metadata) triples for all 3 views per candidate.
    existing = _existing_ids(store)
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    for b in bundles:
        cid = str(b["candidate"]["id"])
        views = render_views(b)
        for view_name, text in views.items():
            doc_id = f"{cid}::{view_name}"
            if doc_id in existing:
                continue
            ids.append(doc_id)
            texts.append(text)
            metadatas.append({"candidate_id": cid, "view": view_name})

    total_to_process = len(texts)
    skipped_existing = len(bundles) * 3 - total_to_process

    log.info(
        "Ingestion plan — candidates=%d, to_process=%d (skipped existing=%d), batch_size=%d",
        len(bundles), total_to_process, skipped_existing, batch_size,
    )

    # Chunk into batches
    batches: list[tuple[list[str], list[str], list[dict]]] = []
    for i in range(0, total_to_process, batch_size):
        batches.append((
            ids[i : i + batch_size],
            texts[i : i + batch_size],
            metadatas[i : i + batch_size],
        ))

    processed = 0
    stopped_reason: str | None = None
    t_start = time.monotonic()

    if total_to_process > 0:
        print(f"Ingesting {total_to_process} docs across {len(batches)} batches "
              f"with {max_workers} parallel workers:", flush=True)

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            executor.submit(_embed_and_upsert_batch, embedder, store, b_ids, b_texts, b_metas): idx
            for idx, (b_ids, b_texts, b_metas) in enumerate(batches)
        }

        for fut in as_completed(futures):
            try:
                n = fut.result()
                processed += n
                _print_progress(processed, total_to_process, t_start)
            except RateLimitError as e:
                stopped_reason = f"rate_limit: {e}"
                log.warning("\nHit rate limit after %d docs — cancelling pending work.", processed)
                for f in futures:
                    f.cancel()
                break
            except APIError as e:
                stopped_reason = f"api_error: {type(e).__name__}: {e}"
                log.warning("\nAPI error after %d docs — cancelling pending work.", processed)
                for f in futures:
                    f.cancel()
                break
    except KeyboardInterrupt:
        stopped_reason = "keyboard_interrupt"
        log.warning("\nInterrupted after %d docs.", processed)
        for f in futures:
            f.cancel()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        if total_to_process > 0:
            print()  # newline after the progress bar

    return {
        "candidates_loaded": len(bundles),
        "documents_to_process": total_to_process,
        "documents_written": processed,
        "documents_skipped_existing": skipped_existing,
        "stopped_reason": stopped_reason,
        "duration_seconds": round(time.monotonic() - t0, 2),
    }
