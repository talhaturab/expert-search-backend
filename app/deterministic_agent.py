"""Deterministic agent — hard filtering + weighted per-dimension scoring + highlights.

No LLM calls. All scoring is SQL-driven compute using the functions in `app.scoring`.
"""
from __future__ import annotations

from app.models import (
    CandidateResult, DimensionSpec, GeoSpec, LanguagesSpec, ParsedSpec,
    SenioritySpec, SkillsSpec,
)
from app.scoring import (
    score_function, score_geography, score_industry,
    score_languages, score_seniority, score_skills,
)

from app.llm_client import client, mod_skills
from app.system_prompt import SKILL_EXPANSION_TEMPLATE


DIM_NAMES = ("function", "industry", "geography", "seniority", "skills", "languages")

# A "required" dim passes if the candidate scores above this threshold. Set just
# below the fuzzy-match cutoff for function/skills so trivial word overlap
# (e.g., "Accountant" vs "Regulatory Affairs" ≈ 0.14) doesn't sneak through.
_HARD_FILTER_THRESHOLD = 0.30


def _dim(spec: ParsedSpec, name: str):
    return getattr(spec, name, None)


def _score_all(bundle: dict, spec: ParsedSpec) -> dict[str, float]:
    """Score every dimension that the spec mentioned. Missing dims are omitted."""
    scores: dict[str, float] = {}
    if spec.function:  scores["function"]  = score_function(bundle, spec.function)
    if spec.industry:  scores["industry"]  = score_industry(bundle, spec.industry)
    if spec.geography: scores["geography"] = score_geography(bundle, spec.geography)
    if spec.seniority: scores["seniority"] = score_seniority(bundle, spec.seniority)
    if spec.skills:    scores["skills"]    = score_skills(bundle, spec.skills)
    if spec.languages: scores["languages"] = score_languages(bundle, spec.languages)
    return scores


def _passes_hard_filters(spec: ParsedSpec, scores: dict[str, float]) -> bool:
    """Required dims must score above _HARD_FILTER_THRESHOLD. No silent degradation."""
    for name in DIM_NAMES:
        d = _dim(spec, name)
        if d is None:
            continue
        required = getattr(d, "required", False)
        if required and scores.get(name, 0.0) < _HARD_FILTER_THRESHOLD:
            return False
    return True


def _weights(spec: ParsedSpec) -> dict[str, float]:
    """Normalized weights per dimension present in the spec.

    Parser weights sum <= 1 by instruction; we re-normalize here so the final
    weighted sum is always in [0, 1]. If all weights are zero but dims are
    mentioned, fall back to uniform.
    """
    w: dict[str, float] = {}
    for name in DIM_NAMES:
        d = _dim(spec, name)
        if d is None:
            continue
        w[name] = getattr(d, "weight", 0.0)
    total = sum(w.values())
    if total > 0:
        return {k: v / total for k, v in w.items()}
    return {k: 1.0 / len(w) for k in w} if w else {}


def build_match_explanation(scores: dict[str, float]) -> str:
    """Human-readable one-sentence rationale from the per-dim scores."""
    if not scores:
        return "No dimensions scored."
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    strong = [f"{name} ({s:.2f})" for name, s in ranked if s >= 0.8][:3]
    weak = [f"{name} ({s:.2f})" for name, s in ranked if s < 0.4][:2]
    parts: list[str] = []
    if strong:
        parts.append("Strong on " + ", ".join(strong))
    if weak:
        parts.append("weak on " + ", ".join(weak))
    return "; ".join(parts) + "." if parts else "Mixed fit across dimensions."


def build_highlights(bundle: dict, spec: ParsedSpec) -> list[str]:
    """2-4 concrete proof-points tied to the spec's dimensions."""
    out: list[str] = []
    c = bundle["candidate"]

    # Industry: current or most recent job matching a target industry
    if spec.industry and spec.industry.values:
        targets = {t.lower() for t in spec.industry.values}
        for w in bundle.get("work", []):
            if (w.get("industry") or "").lower() in targets:
                tag = "current" if w.get("is_current") else "past"
                out.append(
                    f"{tag}: {w.get('job_title', '?')} at {w.get('company', '?')} "
                    f"({w.get('industry', '?')})"
                )
                break

    # Geography: location or nationality call-out
    if spec.geography and spec.geography.values:
        targets = {v.upper() for v in spec.geography.values}
        current = (c.get("country_code") or "").upper()
        if current in targets and c.get("country"):
            out.append(f"Based in {c['country']}")
        elif (c.get("nationality_code") or "").upper() in targets and c.get("nationality"):
            out.append(f"Nationality: {c['nationality']}")

    # Function: best matching job title (keyword search on target)
    if spec.function and spec.function.values:
        targets = [t.lower() for t in spec.function.values]
        for w in bundle.get("work", []):
            title = (w.get("job_title") or "").lower()
            if any(t in title for t in targets):
                yrs = c.get("years_of_experience", "?")
                out.append(f"{yrs}y as {w.get('job_title')}")
                break

    # Skills: up to 2 matching skill names with years
    if spec.skills and spec.skills.values:
        skill_years = {
            (s.get("skill") or "").lower(): s.get("years_of_experience", 0)
            for s in bundle.get("skills", [])
        }
        added = 0
        for t in spec.skills.values:
            if t.lower() in skill_years and added < 2 and len(out) < 4:
                out.append(f"{t} ({skill_years[t.lower()]}y)")
                added += 1

    return out[:4]

from pydantic import BaseModel, Field
from typing import List

class mod_skills(BaseModel):
    skills: List[str] = Field(description="list of application skills")

def get_modified_spec(spec: ParsedSpec) -> ParsedSpec:
    skills = spec.skills.values if spec.skills else []
    response = client.responses.parse(
    model="openai/gpt-5.4",
    input=[
        {
            "role": "system",
            "content": SKILL_EXPANSION_TEMPLATE.render(),
        },
        {
            "role": "user",
            "content": "Expand the following skills into more specific technical skills: " + ", ".join(skills) if skills else "No skills provided.",
        },
    ],
    text_format=mod_skills,
    )
    
    event = response.output_parsed
    modified_skills = event.skills
    
    print('---****---')
    print ("Original skills:", skills)
    print ("Modified skills:", modified_skills)
    print('---****---')
    
    return modified_skills

def filter_and_score(
    bundles: list[dict],
    spec: ParsedSpec,
    top_k: int = 5,
) -> list[CandidateResult]:
    """Main entry point. Hard-filter → score every remaining candidate → top_k."""
    weights = _weights(spec)
    
    # Here we will enrich the spec with LLM-suggested modifications before scoring, if any.
    modified_skills = get_modified_spec(spec)
    spec.skills.values = modified_skills

    scored: list[tuple[float, dict, dict[str, float]]] = []
    for b in bundles:
        scores = _score_all(b, spec)
        if not _passes_hard_filters(spec, scores):
            continue
        final = sum(weights.get(name, 0.0) * scores.get(name, 0.0) for name in scores)
        scored.append((final, b, scores))

    scored.sort(key=lambda t: -t[0])

    results: list[CandidateResult] = []
    for rank, (final, b, scores) in enumerate(scored[:top_k], start=1):
        results.append(CandidateResult(
            candidate_id=str(b["candidate"]["id"]),
            rank=rank,
            score=round(final, 4),
            per_dim={k: round(v, 4) for k, v in scores.items()},
            match_explanation=build_match_explanation(scores),
            highlights=build_highlights(b, spec),
        ))
    return results
