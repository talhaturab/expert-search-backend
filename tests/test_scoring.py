import datetime as dt
import pytest

from app.models import (
    DimensionSpec, GeoSpec, SenioritySpec, SkillsSpec, LanguagesSpec,
)
from app.scoring import (
    score_function, score_industry, score_geography,
    score_seniority, score_skills, score_languages,
)


@pytest.fixture
def bundle_pharma_regulatory_uae():
    return {
        "candidate": {
            "id": "c-1", "first_name": "Ahmed", "last_name": "Hassan",
            "years_of_experience": 12,
            "headline": "Regulatory Affairs Director with deep pharma experience",
            "country_code": "AE", "nationality_code": "AE",
        },
        "work": [
            {"job_title": "Regulatory Affairs Director",
             "start_date": dt.date(2020, 1, 1), "end_date": None, "is_current": True,
             "company": "Pfizer", "industry": "Pharmaceuticals",
             "company_country_code": "AE"},
            {"job_title": "Senior Regulatory Associate",
             "start_date": dt.date(2015, 1, 1), "end_date": dt.date(2020, 1, 1), "is_current": False,
             "company": "Novartis", "industry": "Pharmaceuticals",
             "company_country_code": "CH"},
        ],
        "skills": [
            {"skill": "Regulatory Compliance", "years_of_experience": 10, "proficiency_level": "Expert"},
            {"skill": "FDA Submissions",       "years_of_experience": 7,  "proficiency_level": "Expert"},
        ],
        "languages": [
            {"language": "Arabic",  "proficiency": "Native",   "rank": 4},
            {"language": "English", "proficiency": "Fluent",   "rank": 3},
        ],
    }


def test_score_function_high_for_matching_title(bundle_pharma_regulatory_uae):
    spec = DimensionSpec(values=["Regulatory Affairs"], weight=0.35, required=True)
    score = score_function(bundle_pharma_regulatory_uae, spec)
    assert score > 0.7  # title is a near-exact match on current job


def test_score_industry_full_credit_for_career_in_pharma(bundle_pharma_regulatory_uae):
    spec = DimensionSpec(values=["Pharmaceuticals"], weight=0.30, required=True)
    assert score_industry(bundle_pharma_regulatory_uae, spec) == pytest.approx(1.0, abs=0.01)


def test_score_geography_current_country(bundle_pharma_regulatory_uae):
    spec = GeoSpec(values=["AE", "SA"], weight=0.20, required=False,
                   location_type="current_or_nationality")
    assert score_geography(bundle_pharma_regulatory_uae, spec) == 1.0


def test_score_geography_returns_zero_if_no_match(bundle_pharma_regulatory_uae):
    spec = GeoSpec(values=["US"], weight=0.20, required=False,
                   location_type="current_or_nationality")
    assert score_geography(bundle_pharma_regulatory_uae, spec) == 0.0


def test_score_seniority_maps_12y_to_senior(bundle_pharma_regulatory_uae):
    spec = SenioritySpec(levels=["senior"], weight=0.10, required=False)
    # 12y alone would be "senior" (8-15); "Director" keyword upgrades to "executive".
    # Since target is ["senior"] and boosted level is "executive" (adjacent), score = 0.5.
    assert score_seniority(bundle_pharma_regulatory_uae, spec) == 0.5


def test_score_seniority_executive_for_director_title(bundle_pharma_regulatory_uae):
    spec = SenioritySpec(levels=["executive"], weight=0.10, required=False)
    # Director keyword boosts senior -> executive, which matches target exactly.
    assert score_seniority(bundle_pharma_regulatory_uae, spec) == 1.0


def test_score_skills_hits_over_target(bundle_pharma_regulatory_uae):
    spec = SkillsSpec(values=["Regulatory Compliance", "FDA Submissions", "Russian"],
                      weight=0.10, required=False)
    score = score_skills(bundle_pharma_regulatory_uae, spec)
    assert 0.55 < score < 0.75


def test_score_languages_fraction_matched(bundle_pharma_regulatory_uae):
    spec = LanguagesSpec(values=["Arabic", "French"], required_proficiency=None,
                         weight=0.05, required=False)
    assert score_languages(bundle_pharma_regulatory_uae, spec) == 0.5
