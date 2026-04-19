"""RAG agent — full-scan hybrid retrieval (vector + BM25) with RRF fusion."""
from __future__ import annotations

import numpy as np

from app.bm25_index import BM25Index
from app.models import ViewWeights


VIEW_ORDER: tuple[str, str, str] = ("summary", "work", "skills_edu")


def aggregate_per_candidate(sims_matrix: np.ndarray, view_weights: ViewWeights | None) -> np.ndarray:
    """Collapse a (N_candidates, 3) score matrix into a (N_candidates,) vector.

    Uses query-weighted sum when `view_weights` has any positive component;
    falls back to max otherwise. This matches the spec (§5.2 step 6).
    """
    if view_weights is not None:
        w = np.array(
            [view_weights.summary, view_weights.work, view_weights.skills_edu],
            dtype=np.float32,
        )
        if w.sum() > 0:
            return sims_matrix @ w
    return sims_matrix.max(axis=1)


def rrf_fuse(vec_scores: np.ndarray, bm25_scores: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion of two per-candidate score arrays."""
    def _ranks_desc(a: np.ndarray) -> np.ndarray:
        order = np.argsort(-a)                # indices best-first
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(a))      # rank 0 for best, 1 next, ...
        return ranks
    rv = _ranks_desc(vec_scores)
    rb = _ranks_desc(bm25_scores)
    return 1.0 / (k + rv) + 1.0 / (k + rb)


def retrieve_candidates(
    all_embs: np.ndarray,
    candidate_ids: list[str],
    documents: list[str],
    query_vec: np.ndarray,
    query_text: str,
    view_weights: ViewWeights | None,
    top_k: int = 50,
    bm25_index: BM25Index | None = None,
) -> list[dict]:
    """Full-scan hybrid retrieval. Returns `top_k` candidates by fused score.

    Assumes exactly 3 views per candidate, stored in VIEW_ORDER interleaved:
        [c0_summary, c0_work, c0_skills_edu, c1_summary, c1_work, c1_skills_edu, ...]

    Each result dict:
        {candidate_id, score, rank,
         documents:     {view_name: text},
         vec_per_view:  {view_name: score},
         bm25_per_view: {view_name: score}}
    """
    assert len(all_embs) == len(candidate_ids) == len(documents), \
        "embeddings, ids, and documents must be parallel arrays"

    # 1. Cosine / dot similarity (assumes L2-normalized embeddings; otherwise still ranks OK)
    vec_flat = (all_embs @ query_vec).astype(np.float32)

    # 2. BM25 scores for every document
    if bm25_index is None:
        bm25_index = BM25Index.build(documents)
    bm25_flat = bm25_index.score(query_text)

    # 3. Group by candidate — collapse parallel lists to unique ids preserving order
    unique_ids: list[str] = []
    seen: set[str] = set()
    for cid in candidate_ids:
        if cid not in seen:
            unique_ids.append(cid)
            seen.add(cid)
    n = len(unique_ids)
    assert len(candidate_ids) == n * 3, \
        f"expect exactly 3 views per candidate, got {len(candidate_ids)} for {n} candidates"

    vec_mat = vec_flat.reshape(n, 3)
    bm25_mat = bm25_flat.reshape(n, 3)

    agg_vec = aggregate_per_candidate(vec_mat, view_weights)
    agg_bm25 = aggregate_per_candidate(bm25_mat, view_weights)

    fused = rrf_fuse(agg_vec, agg_bm25)
    top_idx = np.argsort(-fused)[:top_k]

    results: list[dict] = []
    for rank, idx in enumerate(top_idx, start=1):
        base = int(idx) * 3
        results.append({
            "candidate_id": unique_ids[idx],
            "score": float(fused[idx]),
            "rank": rank,
            "documents": {VIEW_ORDER[v]: documents[base + v] for v in range(3)},
            "vec_per_view": {VIEW_ORDER[v]: float(vec_mat[idx, v]) for v in range(3)},
            "bm25_per_view": {VIEW_ORDER[v]: float(bm25_mat[idx, v]) for v in range(3)},
        })
    return results
