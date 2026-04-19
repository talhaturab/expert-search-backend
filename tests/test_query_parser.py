from unittest.mock import MagicMock

from app.models import (
    DimensionSpec, GeoSpec, ParsedSpec, SenioritySpec, ViewWeights,
)
from app.query_parser import parse_query


def test_parse_pharma_middle_east_query_returns_parsed_spec():
    llm = MagicMock()
    llm.chat_structured.return_value = ParsedSpec(
        function=DimensionSpec(values=["Regulatory Affairs"], weight=0.35, required=True),
        industry=DimensionSpec(values=["Pharmaceuticals"], weight=0.30, required=True),
        geography=GeoSpec(values=["AE", "SA"], weight=0.20, required=False,
                          location_type="current_or_nationality"),
        seniority=SenioritySpec(levels=["senior"], weight=0.15, required=False),
        temporality="any",
        view_weights=ViewWeights(summary=0.3, work=0.5, skills_edu=0.2),
    )

    spec = parse_query("regulatory affairs experts in pharma in the Middle East", llm=llm)

    assert isinstance(spec, ParsedSpec)
    assert spec.function is not None and spec.function.required is True
    assert spec.industry is not None and spec.industry.values == ["Pharmaceuticals"]
    assert spec.geography is not None and "AE" in spec.geography.values
    assert spec.view_weights is not None and spec.view_weights.work == 0.5

    # Confirm the LLM was asked for a ParsedSpec
    call_kwargs = llm.chat_structured.call_args.kwargs
    assert call_kwargs["response_model"] is ParsedSpec


def test_parse_empty_query_yields_default_temporality():
    llm = MagicMock()
    llm.chat_structured.return_value = ParsedSpec(temporality="any")
    spec = parse_query("anyone", llm=llm)
    assert spec.temporality == "any"


def test_parse_temporality_past_for_former_role():
    llm = MagicMock()
    llm.chat_structured.return_value = ParsedSpec(
        function=DimensionSpec(values=["CPO"], weight=0.4, required=True),
        temporality="past",
    )
    spec = parse_query("former CPO at petrochemical company", llm=llm)
    assert spec.temporality == "past"
    assert spec.function is not None and spec.function.values == ["CPO"]
