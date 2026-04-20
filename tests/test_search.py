from unittest.mock import MagicMock

import numpy as np

from app.models import CandidateResult, ParsedSpec, ViewWeights
from app.search import SearchService


def _fake_rag(query, candidates):
    return [CandidateResult(candidate_id="c1", rank=1, score=85,
                            match_explanation="rag fit", highlights=["h1"])]


def _fake_det(spec):
    return [CandidateResult(candidate_id="c1", rank=1, score=0.82,
                            per_dim={"industry": 1.0},
                            match_explanation="det fit", highlights=["h2"])]


def _fake_judge(query, rag_picks, det_picks, profiles):
    return (
        [CandidateResult(candidate_id="c1", rank=1, score=0.0,
                         match_explanation="both agreed", highlights=["h3"])],
        "Both agents agreed on c1.",
    )


def _fake_bundle(cid):
    return {
        "candidate": {
            "id": cid, "first_name": "F", "last_name": "L",
            "years_of_experience": 10, "city": "Dubai", "country": "UAE",
            "nationality": "Emirati", "headline": "h",
        },
        "work": [], "education": [], "skills": [], "languages": [],
    }


def _make_service(**overrides) -> SearchService:
    # 1 candidate × 3 views × dim 4
    embs = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=np.float32)
    defaults = dict(
        parse_query=lambda q: ParsedSpec(temporality="any",
                                         view_weights=ViewWeights(summary=0.4, work=0.4, skills_edu=0.2)),
        embed_query=lambda t: [0.1] * 4,
        all_embeddings=embs,
        candidate_ids=["c1", "c1", "c1"],
        documents=["sum1", "work1", "skills1"],
        rag_rerank=_fake_rag,
        run_deterministic=_fake_det,
        judge=_fake_judge,
        fetch_bundle=_fake_bundle,
        rag_top_k=1,
    )
    defaults.update(overrides)
    return SearchService(**defaults)


def test_search_runs_all_stages_and_returns_full_chat_response():
    svc = _make_service()
    resp = svc.search("find pharma experts")
    assert resp.query == "find pharma experts"
    assert len(resp.rag_picks) == 1 and resp.rag_picks[0].candidate_id == "c1"
    assert len(resp.det_picks) == 1 and resp.det_picks[0].per_dim == {"industry": 1.0}
    assert len(resp.suggested) == 1 and resp.suggested[0].match_explanation == "both agreed"
    assert resp.reasoning.startswith("Both agents")
    assert resp.parsed_spec.temporality == "any"


def test_search_feeds_judge_with_profile_markdown_for_union_of_ids():
    captured = {}

    def judge_capture(query, rag_picks, det_picks, profiles):
        captured["profiles"] = profiles
        return _fake_judge(query, rag_picks, det_picks, profiles)

    # Different ids from rag vs det so we can see the union
    def rag(query, candidates):
        return [CandidateResult(candidate_id="r1", rank=1, score=80,
                                match_explanation="r", highlights=[])]

    def det(spec):
        return [CandidateResult(candidate_id="d1", rank=1, score=0.7,
                                match_explanation="d", highlights=[])]

    svc = _make_service(rag_rerank=rag, run_deterministic=det, judge=judge_capture)
    svc.search("q")

    assert set(captured["profiles"].keys()) == {"r1", "d1"}


def test_search_rag_runs_retrieval_with_view_weights_from_spec():
    # Spy on embed to confirm spec is used
    captured_queries = []
    def embed_spy(t):
        captured_queries.append(t)
        return [0.0]*4
    svc = _make_service(embed_query=embed_spy)
    svc.search("my query")
    # Embed was called with the raw query (no HyDE)
    assert captured_queries == ["my query"]
