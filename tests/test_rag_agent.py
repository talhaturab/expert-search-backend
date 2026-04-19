import numpy as np

from app.rag_agent import aggregate_per_candidate, retrieve_candidates, rrf_fuse
from app.models import ViewWeights


def test_aggregate_weighted_sum_when_view_weights_present():
    sims = np.array([
        [0.8, 0.6, 0.4],  # candidate 0
        [0.5, 0.5, 0.5],  # candidate 1
    ])
    weights = ViewWeights(summary=0.5, work=0.3, skills_edu=0.2)
    agg = aggregate_per_candidate(sims, weights)
    # c0: 0.5*0.8 + 0.3*0.6 + 0.2*0.4 = 0.66
    # c1: 0.5*0.5 + 0.3*0.5 + 0.2*0.5 = 0.50
    assert abs(agg[0] - 0.66) < 1e-6
    assert abs(agg[1] - 0.50) < 1e-6


def test_aggregate_falls_back_to_max_when_weights_none():
    sims = np.array([[0.8, 0.6, 0.4], [0.5, 0.9, 0.1]])
    agg = aggregate_per_candidate(sims, None)
    assert abs(agg[0] - 0.8) < 1e-6
    assert abs(agg[1] - 0.9) < 1e-6


def test_aggregate_falls_back_to_max_when_weights_all_zero():
    sims = np.array([[0.8, 0.6, 0.4]])
    agg = aggregate_per_candidate(sims, ViewWeights(summary=0, work=0, skills_edu=0))
    assert abs(agg[0] - 0.8) < 1e-6


def test_rrf_fuse_rewards_candidates_ranked_high_in_both():
    # c0 is rank 0 in both axes; c3 is rank 3 in both.
    vec_scores  = np.array([0.9, 0.8, 0.1, 0.0])  # desc ranks: c0=0, c1=1, c2=2, c3=3
    bm25_scores = np.array([0.95, 0.2, 0.8, 0.0]) # desc ranks: c0=0, c2=1, c1=2, c3=3
    fused = rrf_fuse(vec_scores, bm25_scores, k=60)
    assert np.argmax(fused) == 0  # top in both axes -> strictly highest
    assert np.argmin(fused) == 3  # bottom in both -> strictly lowest


def test_retrieve_candidates_returns_top_k():
    # 3 candidates, 3 views each, embedding dim 4
    all_embs = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],  # c0: summary, work, skills
        [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],  # c1
        [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],  # c2 (orthogonal to query)
    ], dtype=np.float32)
    candidate_ids = ["c0", "c0", "c0", "c1", "c1", "c1", "c2", "c2", "c2"]
    documents = ["c0sum", "c0work", "c0skills",
                 "c1sum", "c1work", "c1skills",
                 "c2sum", "c2work", "c2skills"]

    query_vec = np.array([1, 0, 0, 0], dtype=np.float32)

    results = retrieve_candidates(
        all_embs=all_embs,
        candidate_ids=candidate_ids,
        documents=documents,
        query_vec=query_vec,
        query_text="summary",
        view_weights=None,
        top_k=2,
    )
    assert len(results) == 2
    assert results[0]["candidate_id"] == "c0"   # best match on summary view
    assert "score" in results[0]
    assert "rank" in results[0]
    assert "documents" in results[0]
    assert set(results[0]["documents"].keys()) == {"summary", "work", "skills_edu"}
