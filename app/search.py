"""End-to-end orchestrator. Ties parser + RAG + deterministic + judge into one call.

The service is dependency-injected — every stage is a callable passed in via the
constructor. This makes it trivial to unit-test with mocks and to swap
implementations (e.g., different LLM provider, HyDE toggle) without touching
this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.models import CandidateResult, ChatResponse, ParsedSpec
from app.profile_builder import render_mini
from app.rag_agent import retrieve_candidates


@dataclass
class SearchService:
    # --- stage callables (injected) ---
    parse_query:       Callable[[str], ParsedSpec]
    embed_query:       Callable[[str], list[float]]
    rag_rerank:        Callable[[str, list[dict]], list[CandidateResult]]
    run_deterministic: Callable[[ParsedSpec], list[CandidateResult]]
    judge:             Callable[
        [str, list[CandidateResult], list[CandidateResult], dict[str, str]],
        tuple[list[CandidateResult], str],
    ]
    fetch_bundle:      Callable[[str], dict]

    # --- data loaded at startup ---
    all_embeddings: np.ndarray
    candidate_ids:  list[str]
    documents:      list[str]

    # --- runtime knobs ---
    rag_top_k: int = 50

    def search(self, query: str) -> ChatResponse:
        # 1. Parse the query into a structured spec.
        spec = self.parse_query(query)

        # 2. RAG retrieval — full-scan hybrid with RRF.
        q_list = self.embed_query(query)
        q_vec = np.asarray(q_list, dtype=np.float32)
        pool = retrieve_candidates(
            all_embs=self.all_embeddings,
            candidate_ids=self.candidate_ids,
            documents=self.documents,
            query_vec=q_vec,
            query_text=query,
            view_weights=spec.view_weights,
            top_k=self.rag_top_k,
        )

        # 3. Fan-out to both agents.
        rag_picks = self.rag_rerank(query, pool)
        det_picks = self.run_deterministic(spec)

        # 4. Build mini-profiles for the judge — one per unique candidate_id.
        union_ids = {r.candidate_id for r in rag_picks} | {r.candidate_id for r in det_picks}
        profiles: dict[str, str] = {}
        for cid in union_ids:
            try:
                profiles[cid] = render_mini(self.fetch_bundle(cid))
            except Exception:
                profiles[cid] = "(profile unavailable)"

        suggested, reasoning = self.judge(query, rag_picks, det_picks, profiles)

        return ChatResponse(
            query=query,
            parsed_spec=spec,
            rag_picks=rag_picks,
            det_picks=det_picks,
            suggested=suggested,
            reasoning=reasoning,
        )
