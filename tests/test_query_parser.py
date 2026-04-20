from unittest.mock import MagicMock

from app.models import (
    DimensionSpec, GeoSpec, LanguagesSpec, ParsedSpec, PriorContext,
    SenioritySpec, ViewWeights,
)
from app.query_parser import parse_query
from app.vocabulary import Vocabulary


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


def test_parse_vocab_injection_appends_vocab_block_to_system_prompt():
    llm = MagicMock()
    llm.chat_structured.return_value = ParsedSpec(temporality="any")

    vocab = Vocabulary(
        industries=["Pharmaceuticals"],
        skill_categories=["Engineering"],
        languages=["Arabic"],
        proficiency_levels=["Native"],
        skills=[],
    )
    parse_query("find me X", llm=llm, vocab=vocab)

    system_msg = llm.chat_structured.call_args.kwargs["messages"][0]["content"]
    assert "Known DB vocabulary" in system_msg
    assert "INDUSTRIES: Pharmaceuticals" in system_msg
    assert "LANGUAGES: Arabic" in system_msg


def test_parse_vocab_drops_unknown_industries_and_keeps_known():
    llm = MagicMock()
    llm.chat_structured.return_value = ParsedSpec(
        industry=DimensionSpec(
            values=["Pharmaceuticals", "Made-Up-Industry"], weight=0.3, required=False,
        ),
        languages=LanguagesSpec(
            values=["English", "Klingon"], weight=0.1, required=False,
        ),
        temporality="any",
    )
    vocab = Vocabulary(
        industries=["Pharmaceuticals", "Finance"],
        skill_categories=[],
        languages=["English", "Arabic"],
        proficiency_levels=["Native", "Fluent"],
        skills=[],
    )

    spec = parse_query("x", llm=llm, vocab=vocab)

    assert spec.industry is not None and spec.industry.values == ["Pharmaceuticals"]
    assert spec.languages is not None and spec.languages.values == ["English"]


def test_parse_vocab_canonicalizes_case():
    llm = MagicMock()
    llm.chat_structured.return_value = ParsedSpec(
        industry=DimensionSpec(values=["pharmaceuticals"], weight=0.3, required=False),
        temporality="any",
    )
    vocab = Vocabulary(
        industries=["Pharmaceuticals"], skill_categories=[],
        languages=[], proficiency_levels=[], skills=[],
    )
    spec = parse_query("x", llm=llm, vocab=vocab)
    assert spec.industry is not None and spec.industry.values == ["Pharmaceuticals"]


def test_parse_temporality_past_for_former_role():
    llm = MagicMock()
    llm.chat_structured.return_value = ParsedSpec(
        function=DimensionSpec(values=["CPO"], weight=0.4, required=True),
        temporality="past",
    )
    spec = parse_query("former CPO at petrochemical company", llm=llm)
    assert spec.temporality == "past"
    assert spec.function is not None and spec.function.values == ["CPO"]


def test_parser_without_prior_context_does_not_mention_refinement_block():
    llm = MagicMock()
    llm.chat_structured.return_value = ParsedSpec(temporality="any")
    parse_query("find pharma experts", llm=llm)
    system_msg = llm.chat_structured.call_args.kwargs["messages"][0]["content"]
    assert "PRIOR TURN CONTEXT" not in system_msg
    assert "REFINEMENT DETECTION" not in system_msg


def test_parser_with_prior_context_injects_block_with_ids_and_prior_query():
    llm = MagicMock()
    llm.chat_structured.return_value = ParsedSpec(temporality="any", is_refinement=True)
    prior = PriorContext(
        prior_query="regulatory affairs in pharma in Middle East",
        prior_parsed_spec=ParsedSpec(temporality="any"),
        prior_suggested_ids=["abc-123", "def-456"],
    )
    spec = parse_query("filter those to only people in Saudi Arabia", llm=llm, prior_context=prior)
    assert spec.is_refinement is True
    system_msg = llm.chat_structured.call_args.kwargs["messages"][0]["content"]
    assert "PRIOR TURN CONTEXT" in system_msg
    assert "REFINEMENT DETECTION" in system_msg
    assert "regulatory affairs in pharma in Middle East" in system_msg
    assert "abc-123" in system_msg
