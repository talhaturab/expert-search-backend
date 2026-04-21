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


def test_score_skills_fuzzy_match_picks_up_variant_spelling(bundle_pharma_regulatory_uae):
    # Candidate has "Regulatory Compliance"; query says "Regulatory compliances" (variant)
    spec = SkillsSpec(values=["Regulatory compliances"], weight=0.1, required=False)
    score = score_skills(bundle_pharma_regulatory_uae, spec)
    assert score > 0.0  # fuzzy match should succeed


def test_score_skills_no_match_returns_zero(bundle_pharma_regulatory_uae):
    spec = SkillsSpec(values=["Quantum Computing"], weight=0.1, required=False)
    assert score_skills(bundle_pharma_regulatory_uae, spec) == 0.0


def test_score_skills_short_target_matches_longer_candidate_skill():
    """Short LLM target ("Security") should match a longer candidate skill
    ("Security Assurance") via word-subset. Trigram similarity (0.62) alone
    fails against the 0.80 threshold."""
    bundle = {
        "candidate": {"id": "c-sec", "first_name": "Hassan", "last_name": "Patel"},
        "work": [],
        "skills": [
            {"skill": "Security Assurance", "years_of_experience": 12, "proficiency_level": "Beginner"},
        ],
    }
    spec = SkillsSpec(values=["Security"], weight=0.2, required=False)
    assert score_skills(bundle, spec) > 0.6


def test_score_skills_word_subset_does_not_create_false_positives():
    """"AI" should NOT match "AIDS" — word-subset only accepts whole-word
    containment, so "AI" ⊂ "AIDS" is false."""
    bundle = {
        "candidate": {"id": "c-ai", "first_name": "x", "last_name": "y"},
        "work": [],
        "skills": [{"skill": "AIDS", "years_of_experience": 1, "proficiency_level": "Beginner"}],
    }
    spec = SkillsSpec(values=["AI"], weight=0.1, required=False)
    assert score_skills(bundle, spec) == 0.0


def test_score_industry_current_role_floor_for_career_switcher():
    """Someone currently in a target industry should score >= 0.7 even if most
    of their career was elsewhere. This is the "find me someone at a pharma
    company right now" case — recency > lifetime tenure."""
    bundle = {
        "candidate": {"id": "c-switch", "first_name": "x", "last_name": "y",
                      "years_of_experience": 15},
        "work": [
            # 2 years in target industry (current)
            {"job_title": "Data Scientist", "is_current": True,
             "start_date": dt.date(2024, 1, 1), "end_date": None,
             "company": "Pfizer", "industry": "Pharmaceuticals",
             "company_country_code": "US"},
            # 13 years in a different industry
            {"job_title": "Engineer", "is_current": False,
             "start_date": dt.date(2011, 1, 1), "end_date": dt.date(2024, 1, 1),
             "company": "Ford", "industry": "Automotive",
             "company_country_code": "US"},
        ],
        "skills": [], "languages": [],
    }
    spec = DimensionSpec(values=["Pharmaceuticals"], weight=0.3, required=False)
    score = score_industry(bundle, spec)
    # Old behaviour: 2/15 ≈ 0.13. New behaviour: max(0.7, 0.13) = 0.7.
    assert score >= 0.7


def test_score_industry_full_tenure_still_scores_one():
    """Regression: lifetime pharma veteran must still score 1.0, not be
    capped by the current-role floor."""
    bundle = {
        "candidate": {"id": "c-vet", "first_name": "x", "last_name": "y",
                      "years_of_experience": 20},
        "work": [
            {"job_title": "Director", "is_current": True,
             "start_date": dt.date(2010, 1, 1), "end_date": None,
             "company": "Pfizer", "industry": "Pharmaceuticals",
             "company_country_code": "US"},
        ],
        "skills": [], "languages": [],
    }
    spec = DimensionSpec(values=["Pharmaceuticals"], weight=0.3, required=False)
    assert score_industry(bundle, spec) == pytest.approx(1.0, abs=0.01)


def test_score_industry_not_currently_in_target_uses_tenure():
    """Not currently in target industry → pure tenure fraction, no floor."""
    bundle = {
        "candidate": {"id": "c-ex", "first_name": "x", "last_name": "y",
                      "years_of_experience": 10},
        "work": [
            # 2 years in non-target industry (current)
            {"job_title": "Engineer", "is_current": True,
             "start_date": dt.date(2024, 1, 1), "end_date": None,
             "company": "Ford", "industry": "Automotive",
             "company_country_code": "US"},
            # 8 years in target industry (past)
            {"job_title": "Scientist", "is_current": False,
             "start_date": dt.date(2016, 1, 1), "end_date": dt.date(2024, 1, 1),
             "company": "Pfizer", "industry": "Pharmaceuticals",
             "company_country_code": "US"},
        ],
        "skills": [], "languages": [],
    }
    spec = DimensionSpec(values=["Pharmaceuticals"], weight=0.3, required=False)
    # 8/10 = 0.8 tenure, no current-role floor → 0.8
    assert 0.75 < score_industry(bundle, spec) < 0.85


def test_score_languages_fraction_matched(bundle_pharma_regulatory_uae):
    spec = LanguagesSpec(values=["Arabic", "French"], required_proficiency=None,
                         weight=0.05, required=False)
    assert score_languages(bundle_pharma_regulatory_uae, spec) == 0.5
