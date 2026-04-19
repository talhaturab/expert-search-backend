from typing import Literal
from pydantic import BaseModel, Field


# ---------- Parsed Spec (query parser output) ----------

class DimensionSpec(BaseModel):
    values: list[str] = Field(default_factory=list)
    weight: float = 0.0
    required: bool = False


class GeoSpec(DimensionSpec):
    location_type: Literal["current", "current_or_nationality", "historical"] = "current_or_nationality"


class SenioritySpec(BaseModel):
    levels: list[Literal["junior", "mid", "senior", "executive"]] = Field(default_factory=list)
    weight: float = 0.0
    required: bool = False


class SkillsSpec(BaseModel):
    values: list[str] = Field(default_factory=list)
    weight: float = 0.0
    required: bool = False


class LanguagesSpec(BaseModel):
    values: list[str] = Field(default_factory=list)
    required_proficiency: Literal["Beginner", "Intermediate", "Fluent", "Native"] | None = None
    weight: float = 0.0
    required: bool = False


ViewName = Literal["summary", "work", "skills_edu"]


class ViewWeights(BaseModel):
    """Per-view aggregation weights for RAG. Avoid `dict[Literal, float]` because
    it emits `propertyNames` in JSON Schema, which Anthropic's validator rejects."""
    summary: float = 0.0
    work: float = 0.0
    skills_edu: float = 0.0


class ParsedSpec(BaseModel):
    function: DimensionSpec | None = None
    industry: DimensionSpec | None = None
    geography: GeoSpec | None = None
    seniority: SenioritySpec | None = None
    skills: SkillsSpec | None = None
    languages: LanguagesSpec | None = None
    min_years_exp: int | None = None
    temporality: Literal["current", "past", "any"] = "any"
    view_weights: ViewWeights | None = None


# ---------- Result structures ----------

class CandidateResult(BaseModel):
    candidate_id: str
    rank: int
    score: float
    match_explanation: str
    highlights: list[str] = Field(default_factory=list)
    per_dim: dict[str, float] | None = None  # populated only for det_picks


class RerankPick(BaseModel):
    """Structured LLM output for a single-candidate pointwise rerank call.

    Note: `score` is intended to be 0-100 but we don't express ge/le as JSON
    Schema constraints — Anthropic rejects `minimum`/`maximum` on `integer`.
    We clamp in code after parsing instead.
    """
    score: int = Field(description="0-100 fit score (100=perfect, 0=off-topic)")
    match_explanation: str = Field(description="One sentence on why this score")
    highlights: list[str] = Field(default_factory=list,
                                  description="2-4 concrete proof-points from the profile")


# ---------- Request / Response models ----------

class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None  # no-op; reserved for future use


class ChatResponse(BaseModel):
    query: str
    parsed_spec: ParsedSpec
    rag_picks: list[CandidateResult]
    det_picks: list[CandidateResult]
    suggested: list[CandidateResult]
    reasoning: str


class IngestRequest(BaseModel):
    force: bool = False  # re-embed even if the Chroma collection already exists


class IngestResponse(BaseModel):
    candidates_indexed: int
    documents_written: int
    duration_seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, bool]
