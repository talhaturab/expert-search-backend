import pytest
from pydantic import ValidationError

from app.models import (
    DimensionSpec, GeoSpec, SenioritySpec, SkillsSpec, LanguagesSpec,
    ParsedSpec, CandidateResult, ChatRequest, ChatResponse,
    IngestRequest, HealthResponse,
)


def test_parsed_spec_minimal():
    spec = ParsedSpec(temporality="any")
    assert spec.function is None
    assert spec.industry is None
    assert spec.temporality == "any"
    assert spec.view_weights is None


def test_parsed_spec_full():
    spec = ParsedSpec(
        function=DimensionSpec(values=["Regulatory Affairs"], weight=0.35, required=True),
        industry=DimensionSpec(values=["Pharmaceuticals"], weight=0.30, required=True),
        geography=GeoSpec(values=["AE", "SA"], weight=0.20, required=False,
                          location_type="current_or_nationality"),
        seniority=SenioritySpec(levels=["senior"], weight=0.10, required=False),
        temporality="any",
        view_weights={"summary": 0.3, "work": 0.5, "skills_edu": 0.2},
    )
    assert spec.function.values == ["Regulatory Affairs"]
    assert spec.geography.location_type == "current_or_nationality"


def test_parsed_spec_invalid_temporality():
    with pytest.raises(ValidationError):
        ParsedSpec(temporality="future")


def test_candidate_result():
    r = CandidateResult(
        candidate_id="c-1", rank=1, score=0.82,
        match_explanation="Strong industry match.",
        highlights=["12y pharma", "based in Dubai"],
    )
    assert r.rank == 1
    assert r.per_dim is None


def test_chat_request_default_conversation_id_is_none():
    req = ChatRequest(query="find pharma experts")
    assert req.conversation_id is None


def test_chat_response_shape():
    resp = ChatResponse(
        query="q",
        parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[],
        det_picks=[],
        suggested=[],
        reasoning="empty",
    )
    assert resp.query == "q"
