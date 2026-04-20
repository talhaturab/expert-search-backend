"""POST /chat — the user-facing expert-search endpoint.

The heavy lifting happens in `get_search_service()` which builds (and caches)
the fully-wired SearchService. First call loads Chroma + DB bundles + vocab;
subsequent calls reuse the cached service. After /ingest succeeds the cache
should be invalidated so the next /chat reloads the fresh Chroma collection.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.chroma_store import ChromaStore
from app.config import get_settings
from app.db import fetch_all_bundles, fetch_candidate_bundle
from app.deterministic_agent import filter_and_score
from app.embeddings import EmbeddingClient
from app.judge import cherry_pick_top_five
from app.llm import LLMClient
from app.models import ChatRequest, ChatResponse
from app.query_parser import parse_query
from app.rag_agent import rerank_and_explain
from app.search import SearchService
from app.vocabulary import load_vocabulary


log = logging.getLogger(__name__)

router = APIRouter()


@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    """Build a fully-wired SearchService once per process.

    Loads Chroma embeddings, candidate bundles, vocabulary. Expensive; cached.
    Call `invalidate_search_service()` after /ingest rebuilds the index.
    """
    s = get_settings()

    # Load Chroma into memory — this is the numpy matrix we full-scan at query time.
    log.info("Loading Chroma embeddings...")
    store = ChromaStore(persist_path=s.chroma_persist_path)
    embs, metas, docs, _ = store.load_all()
    if len(embs) == 0:
        raise HTTPException(
            status_code=503,
            detail="Chroma is empty. POST /ingest to build the index first.",
        )
    candidate_ids_in_order = [m["candidate_id"] for m in metas]

    # Load bundles for every candidate the index references.
    log.info("Loading candidate bundles from Postgres...")
    all_bundles = fetch_all_bundles(s.database_url, limit=s.ingest_limit)
    bundles_by_id = {str(b["candidate"]["id"]): b for b in all_bundles}

    # Load DB vocabulary once.
    log.info("Loading DB vocabulary...")
    vocab = load_vocabulary(s.database_url)

    # Clients.
    llm = LLMClient(api_key=s.openrouter_api_key, model=s.llm_model)
    embedder = EmbeddingClient(api_key=s.openrouter_api_key, model=s.embedding_model)

    def _fetch_bundle(cid: str) -> dict:
        b = bundles_by_id.get(cid)
        if b is not None:
            return b
        # Fallback — the judge may receive an id that isn't in the indexed set.
        return fetch_candidate_bundle(s.database_url, cid)

    return SearchService(
        parse_query=lambda q: parse_query(q, llm=llm, vocab=vocab),
        embed_query=lambda t: embedder.embed_one(t),
        all_embeddings=embs,
        candidate_ids=candidate_ids_in_order,
        documents=docs,
        rag_rerank=lambda q, pool: rerank_and_explain(
            q, pool, llm=llm, top_k=s.final_top_k
        ),
        run_deterministic=lambda spec: filter_and_score(
            all_bundles, spec, top_k=s.deterministic_top_k
        ),
        judge=lambda q, rp, dp, pm: cherry_pick_top_five(q, rp, dp, pm, llm=llm),
        fetch_bundle=_fetch_bundle,
        rag_top_k=s.rag_top_k,
    )


def invalidate_search_service() -> None:
    """Clear the cached service — call after /ingest succeeds."""
    get_search_service.cache_clear()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    service = get_search_service()
    return service.search(req.query)
