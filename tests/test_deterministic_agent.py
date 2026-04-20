import datetime as dt
import pytest

from app.deterministic_agent import (
    build_highlights,
    build_match_explanation,
    filter_and_score,
)
from app.models import (
    DimensionSpec, GeoSpec, ParsedSpec, SenioritySpec,
)


def _bundle(cid: str, industry: str, country: str, years: int, title: str,
            is_current: bool = True) -> dict:
    return {
        "candidate": {
            "id": cid, "first_name": "F", "last_name": "L",
            "country_code": country, "nationality_code": country,
            "country": country, "nationality": country,
            "years_of_experience": years, "headline": title,
        },
        "work": [
            {"job_title": title, "start_date": dt.date(2015, 1, 1),
             "end_date": None if is_current else dt.date(2024, 1, 1),
             "is_current": is_current,
             "company": "X", "industry": industry,
             "company_country_code": country, "company_country": country},
        ],
        "education": [],
        "skills": [],
        "languages": [],
    }


@pytest.fixture
def bundles():
    # "Regulatory Affairs Director" used as title to satisfy both function AND seniority.
    return [
        _bundle("c-match",     "Pharmaceuticals", "AE", 12, "Regulatory Affairs Director"),
        _bundle("c-wrongind",  "Finance",         "AE", 12, "Regulatory Affairs Director"),
        _bundle("c-wronggeo",  "Pharmaceuticals", "US", 12, "Regulatory Affairs Director"),
        _bundle("c-wrongrole", "Pharmaceuticals", "AE", 12, "Accountant"),
    ]


@pytest.fixture
def pharma_regulatory_me_spec():
    return ParsedSpec(
        function=DimensionSpec(values=["Regulatory Affairs"], weight=0.35, required=True),
        industry=DimensionSpec(values=["Pharmaceuticals"], weight=0.30, required=True),
        geography=GeoSpec(values=["AE", "SA", "QA"], weight=0.20, required=False,
                          location_type="current_or_nationality"),
        seniority=SenioritySpec(levels=["senior"], weight=0.15, required=False),
        temporality="any",
    )


def test_hard_filters_exclude_non_pharma_and_wrong_role(bundles, pharma_regulatory_me_spec):
    picks = filter_and_score(bundles, pharma_regulatory_me_spec, top_k=5)
    picked_ids = {p.candidate_id for p in picks}
    assert "c-wrongind" not in picked_ids     # fails industry hard filter
    assert "c-wrongrole" not in picked_ids    # fails function hard filter
    # c-match and c-wronggeo should survive
    assert "c-match" in picked_ids


def test_top_match_is_exact_fit(bundles, pharma_regulatory_me_spec):
    picks = filter_and_score(bundles, pharma_regulatory_me_spec, top_k=5)
    assert picks[0].candidate_id == "c-match"
    assert picks[0].per_dim is not None
    assert picks[0].per_dim["industry"] == pytest.approx(1.0, abs=0.01)
    assert picks[0].per_dim["geography"] == 1.0


def test_hard_filter_empty_pool_returns_empty(pharma_regulatory_me_spec):
    picks = filter_and_score(
        [_bundle("only-finance", "Finance", "US", 12, "Accountant")],
        pharma_regulatory_me_spec, top_k=5,
    )
    assert picks == []


def test_match_explanation_mentions_strong_dims():
    explanation = build_match_explanation(
        {"function": 0.91, "industry": 1.0, "geography": 1.0, "seniority": 0.5}
    )
    # Strong dims (>= 0.8) should be referenced
    assert "industry" in explanation.lower() or "1.00" in explanation


def test_build_highlights_picks_matching_industry_job(bundles, pharma_regulatory_me_spec):
    highlights = build_highlights(bundles[0], pharma_regulatory_me_spec)
    assert any(
        "Pharmaceuticals" in h or "Regulatory" in h for h in highlights
    )


def test_filter_and_score_sorts_by_total_weighted_sum(pharma_regulatory_me_spec):
    # Both pass hard filters (Pharma industry + Regulatory title),
    # but only c-good is in target geography → higher total.
    candidates = [
        _bundle("c-good",  "Pharmaceuticals", "AE", 12, "Regulatory Affairs Director"),
        _bundle("c-okay",  "Pharmaceuticals", "US", 12, "Regulatory Affairs Director"),
    ]
    picks = filter_and_score(candidates, pharma_regulatory_me_spec, top_k=5)
    assert picks[0].candidate_id == "c-good"
    assert picks[0].score > picks[1].score
