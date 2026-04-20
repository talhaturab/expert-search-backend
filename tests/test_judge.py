from unittest.mock import MagicMock

from app.judge import cherry_pick_top_five
from app.models import CandidateResult, JudgeDecision, JudgePick


def _cr(cid: str, rank: int, score: float, per_dim: dict | None = None) -> CandidateResult:
    return CandidateResult(
        candidate_id=cid, rank=rank, score=score,
        match_explanation=f"{cid} explanation",
        highlights=[f"{cid} h1", f"{cid} h2"],
        per_dim=per_dim,
    )


def test_judge_returns_5_ranked_candidates_with_reasoning():
    llm = MagicMock()
    llm.chat_structured.return_value = JudgeDecision(
        suggested=[
            JudgePick(candidate_id="a", match_explanation="both lists agreed", highlights=["aa"]),
            JudgePick(candidate_id="b", match_explanation="strong semantic match", highlights=["bb"]),
            JudgePick(candidate_id="c", match_explanation="deep tenure", highlights=["cc"]),
            JudgePick(candidate_id="d", match_explanation="matches geography + industry", highlights=["dd"]),
            JudgePick(candidate_id="e", match_explanation="title keyword match", highlights=["ee"]),
        ],
        reasoning="Candidates a, b appeared in both lists; c/d/e from deterministic only.",
    )

    rag = [_cr("a", 1, 85), _cr("b", 2, 80), _cr("f", 3, 70), _cr("g", 4, 60), _cr("h", 5, 55)]
    det = [_cr("a", 1, 0.92, {"industry": 1.0}),
           _cr("c", 2, 0.88, {"industry": 1.0}),
           _cr("d", 3, 0.78, {"industry": 0.8}),
           _cr("e", 4, 0.70, {"industry": 0.7}),
           _cr("i", 5, 0.60, {"industry": 0.5})]
    profiles = {c: f"profile md for {c}" for c in "abcdefghi"}

    suggested, reasoning = cherry_pick_top_five(
        query="regulatory pharma ME", rag_picks=rag, det_picks=det,
        profile_markdown=profiles, llm=llm,
    )

    assert len(suggested) == 5
    assert [p.candidate_id for p in suggested] == ["a", "b", "c", "d", "e"]
    assert [p.rank for p in suggested] == [1, 2, 3, 4, 5]
    assert suggested[0].match_explanation == "both lists agreed"
    assert "both lists" in reasoning.lower()

    # Confirm chat_structured was called with the right response model
    kwargs = llm.chat_structured.call_args.kwargs
    assert kwargs["response_model"] is JudgeDecision


def test_judge_truncates_to_5_if_llm_returns_more():
    llm = MagicMock()
    llm.chat_structured.return_value = JudgeDecision(
        suggested=[JudgePick(candidate_id=str(i), match_explanation=f"m{i}")
                   for i in range(8)],
        reasoning="I ignored the limit",
    )
    picks, _ = cherry_pick_top_five(
        query="q", rag_picks=[_cr("0", 1, 0.9)], det_picks=[_cr("1", 1, 0.9)],
        profile_markdown={str(i): "..." for i in range(8)}, llm=llm,
    )
    assert len(picks) == 5


def test_judge_passes_both_lists_and_profiles_in_prompt():
    llm = MagicMock()
    llm.chat_structured.return_value = JudgeDecision(
        suggested=[JudgePick(candidate_id="x", match_explanation="ok")],
        reasoning="short",
    )
    cherry_pick_top_five(
        query="find X",
        rag_picks=[_cr("x", 1, 75)],
        det_picks=[_cr("y", 1, 0.82, {"industry": 1.0})],
        profile_markdown={"x": "PROFILE_X_MARKDOWN", "y": "PROFILE_Y_MARKDOWN"},
        llm=llm,
    )
    user = llm.chat_structured.call_args.kwargs["messages"][-1]["content"]
    # Judge sees both labels, both profiles, and the per-dim breakdown
    assert "RAG" in user and "Deterministic" in user
    assert "PROFILE_X_MARKDOWN" in user and "PROFILE_Y_MARKDOWN" in user
    assert "industry=1.00" in user
