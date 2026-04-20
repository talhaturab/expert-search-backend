"""End-to-end orchestrator. Ties parser + RAG + deterministic + judge into one call.

Dependency-injected — every stage is a callable passed in via the constructor.
Refinement: if the caller supplies a `conversation_id` whose last turn exists
and the parser classifies the new query as `is_refinement=True`, we restrict
the search to the prior turn's suggested candidates instead of running a fresh
full-scan retrieval.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.models import (
    CandidateResult, ChatResponse, ParsedSpec, PriorContext,
)
from app.profile_builder import render_mini
from app.rag_agent import retrieve_candidates
from app.session_store import SessionStore, SessionTurn


def _new_conversation_id() -> str:
    return uuid.uuid4().hex


@dataclass
class SearchService:
    parse_query:       Callable[..., ParsedSpec]
    embed_query:       Callable[[str], list[float]]
    rag_rerank:        Callable[[str, list[dict]], list[CandidateResult]]
    run_deterministic: Callable[[ParsedSpec], list[CandidateResult]]
    run_deterministic_on_pool: Callable[[list[dict], ParsedSpec, int], list[CandidateResult]]
    judge:             Callable[
        [str, list[CandidateResult], list[CandidateResult], dict[str, str]],
        tuple[list[CandidateResult], str],
    ]
    fetch_bundle:      Callable[[str], dict]

    all_embeddings: np.ndarray
    candidate_ids:  list[str]
    documents:      list[str]

    rag_top_k: int = 50

    session_store: SessionStore = field(default_factory=lambda: SessionStore(ttl_seconds=1800))

    def search(self, query: str, conversation_id: str | None = None) -> ChatResponse:
        cid = conversation_id or _new_conversation_id()

        prior_turn = self.session_store.get(cid)
        prior_ctx = None
        if prior_turn and prior_turn.response.suggested:
            prior_ctx = PriorContext(
                prior_query=prior_turn.query,
                prior_parsed_spec=prior_turn.response.parsed_spec,
                prior_suggested_ids=[c.candidate_id for c in prior_turn.response.suggested],
            )

        spec = self.parse_query(query, prior_context=prior_ctx)

        is_refinement = (
            prior_ctx is not None
            and spec.is_refinement
            and len(prior_ctx.prior_suggested_ids) > 0
        )

        if is_refinement:
            response = self._refined_search(query, spec, prior_ctx)
        else:
            response = self._fresh_search(query, spec)

        response.conversation_id = cid
        response.is_refinement = is_refinement
        self.session_store.put(
            cid,
            SessionTurn(query=query, response=response, timestamp=time.monotonic()),
        )
        return response

    def _fresh_search(self, query: str, spec: ParsedSpec) -> ChatResponse:
        q_vec = np.asarray(self.embed_query(query), dtype=np.float32)
        pool = retrieve_candidates(
            all_embs=self.all_embeddings,
            candidate_ids=self.candidate_ids,
            documents=self.documents,
            query_vec=q_vec,
            query_text=query,
            view_weights=spec.view_weights,
            top_k=self.rag_top_k,
        )
        rag_picks = self.rag_rerank(query, pool)
        det_picks = self.run_deterministic(spec)
        return self._finalize(query, spec, rag_picks, det_picks)

    def _refined_search(
        self, query: str, spec: ParsedSpec, prior_ctx: PriorContext
    ) -> ChatResponse:
        pool_bundles: list[dict] = []
        for cid_ in prior_ctx.prior_suggested_ids:
            try:
                pool_bundles.append(self.fetch_bundle(cid_))
            except Exception:
                pass

        det_picks = self.run_deterministic_on_pool(pool_bundles, spec, 5)

        rag_input = [
            {
                "candidate_id": str(b["candidate"]["id"]),
                "score": 0.0,
                "rank": i + 1,
                "documents": {
                    "summary":    render_mini(b),
                    "work":       "",
                    "skills_edu": "",
                },
            }
            for i, b in enumerate(pool_bundles)
        ]
        rag_picks = self.rag_rerank(query, rag_input) if rag_input else []

        # Reuse the bundles we already fetched for the restricted pool so we
        # don't call `fetch_bundle` twice per candidate during refinement.
        prefetched = {str(b["candidate"]["id"]): b for b in pool_bundles}

        response = self._finalize(query, spec, rag_picks, det_picks, prefetched)
        if not response.suggested:
            response.reasoning = (
                "No prior candidates matched the new constraint — "
                "try a broader follow-up or a fresh query."
            )
        return response

    def _finalize(
        self,
        query: str,
        spec: ParsedSpec,
        rag_picks: list[CandidateResult],
        det_picks: list[CandidateResult],
        prefetched_bundles: dict[str, dict] | None = None,
    ) -> ChatResponse:
        union_ids = (
            {r.candidate_id for r in rag_picks}
            | {r.candidate_id for r in det_picks}
        )
        profiles: dict[str, str] = {}
        for cid in union_ids:
            try:
                if prefetched_bundles is not None and cid in prefetched_bundles:
                    profiles[cid] = render_mini(prefetched_bundles[cid])
                else:
                    profiles[cid] = render_mini(self.fetch_bundle(cid))
            except Exception:
                profiles[cid] = "(profile unavailable)"

        if rag_picks or det_picks:
            suggested, reasoning = self.judge(query, rag_picks, det_picks, profiles)
        else:
            suggested, reasoning = [], ""

        # `conversation_id` and `is_refinement` are intentionally placeholders
        # here — `search()` stamps them on the returned response (see lines 76-77).
        # Do not call `_finalize` directly and expect those fields populated.
        return ChatResponse(
            query=query,
            conversation_id="",
            is_refinement=False,
            parsed_spec=spec,
            rag_picks=rag_picks,
            det_picks=det_picks,
            suggested=suggested,
            reasoning=reasoning,
        )
