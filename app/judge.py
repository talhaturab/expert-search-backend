"""Judge — LLM cherry-picks the final top-5 across both agents' outputs.

Single LLM call via structured outputs. The judge sees:
- the original query,
- the RAG agent's top-5 with per-candidate explanations,
- the deterministic agent's top-5 with per-dim score breakdowns,
- a full mini-profile for every unique candidate across both lists.

It returns the 5 candidates it thinks best answer the query, plus a short
reasoning string explaining the blend.
"""
from __future__ import annotations

from app.llm import LLMClient
from app.models import CandidateResult, JudgeDecision


SYSTEM_PROMPT = """You are a senior associate choosing which 5 candidates to present for a
natural-language expert-search query. You have two ranked lists from upstream agents:

1. RAG (semantic retrieval + parallel pointwise rerank). Scores are 0-100
   fit judgments from the reranker.
2. Deterministic (structured SQL scoring across function/industry/geography/
   seniority/skills/languages). Scores are normalized weighted sums in [0, 1],
   and each candidate carries a per-dimension breakdown.

Your job: cherry-pick the 5 candidates most relevant to the query. You may
draw from either list in any mix. The goal is the 5 candidates who most truly
fit — not which agent was "right."

Rules:
- Output exactly 5 candidates. If fewer than 5 distinct candidates are
  available across both lists, repeat nothing — return however many exist.
- Each candidate_id in your output MUST be one that appeared in one of the
  two input lists.
- Each match_explanation is ONE short sentence explaining why this candidate
  made it into the final 5, including a signal from at least one agent.
- Each highlights entry is a concrete proof-point (e.g., "12y at Pfizer",
  "based in Dubai") — not vague praise.
- reasoning is 2-4 sentences on your overall decisions: where the agents
  agreed, where you favored one over the other, what you weighted and why."""


def _fmt_list(label: str, results: list[CandidateResult], profiles: dict[str, str]) -> str:
    if not results:
        return f"=== {label} ===\n(no results)"
    lines = [f"=== {label} ==="]
    for r in results:
        per_dim = ""
        if r.per_dim:
            per_dim = " | per_dim=" + ", ".join(f"{k}={v:.2f}" for k, v in r.per_dim.items())
        profile = profiles.get(r.candidate_id, "(profile unavailable)")
        lines.append(
            f"[rank {r.rank}] candidate_id={r.candidate_id} score={r.score:.4f}{per_dim}\n"
            f"  Agent note: {r.match_explanation}\n"
            f"  Profile:    {profile}"
        )
    return "\n".join(lines)


def cherry_pick_top_five(
    query: str,
    rag_picks: list[CandidateResult],
    det_picks: list[CandidateResult],
    profile_markdown: dict[str, str],
    llm: LLMClient,
) -> tuple[list[CandidateResult], str]:
    """Returns (suggested CandidateResult list, reasoning string)."""
    user = (
        f"Query:\n{query}\n\n"
        f"{_fmt_list('RAG (semantic)', rag_picks, profile_markdown)}\n\n"
        f"{_fmt_list('Deterministic (structured)', det_picks, profile_markdown)}\n\n"
        "Return your JudgeDecision now."
    )

    decision = llm.chat_structured(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ],
        response_model=JudgeDecision,
        temperature=0.0,
    )

    suggested: list[CandidateResult] = []
    for rank, pick in enumerate(decision.suggested[:5], start=1):
        suggested.append(CandidateResult(
            candidate_id=pick.candidate_id,
            rank=rank,
            score=0.0,  # judge doesn't emit a numeric score; caller can cross-reference
            match_explanation=pick.match_explanation,
            highlights=list(pick.highlights),
        ))
    return suggested, decision.reasoning
