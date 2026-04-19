import datetime as dt

import pytest
from app.config import Settings


@pytest.fixture
def settings(monkeypatch, tmp_path) -> Settings:
    """Minimal valid settings for unit tests."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://developer:devread2024@34.79.32.228:5432/candidate_profiles")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))
    return Settings()


@pytest.fixture
def real_db_settings(settings):
    """Alias — use this fixture when the test hits the real DB (read-only)."""
    return settings


@pytest.fixture
def sample_bundle():
    """Hardcoded Sara-Ali-like bundle for unit tests (no DB required)."""
    return {
        "candidate": {
            "id": "cid-001",
            "first_name": "Sara",
            "last_name": "Ali",
            "email": "saraali@hotmail.com",
            "phone": "+97323374090",
            "date_of_birth": dt.date(1980, 1, 4),
            "gender": "Female",
            "headline": "Results-driven Engineering Manager with expertise in Brand Marketing",
            "years_of_experience": 16,
            "city": "Philadelphia",
            "country": "United States",
            "country_code": "US",
            "nationality": "Latvia",
            "nationality_code": "LV",
            "created_at": dt.datetime(2026, 2, 20, 7, 38, 21),
        },
        "work": [
            {"job_title": "Engineering Manager Research Scientist",
             "start_date": dt.date(2025, 4, 1), "end_date": None,
             "is_current": True, "description": "Spearheaded migration to analysis, reducing costs by 58%.",
             "company": "IDM Brokerage House", "industry": "Finance",
             "company_country": "Czech Republic", "company_country_code": "CZ"},
        ],
        "education": [
            {"start_year": 2007, "graduation_year": 2010, "grade": "B-",
             "institution": "Union High School", "degree": "Associate's",
             "field": "Petroleum and Natural Gas Engineering"},
        ],
        "skills": [
            {"skill": "Brand Marketing", "category": "Marketing",
             "years_of_experience": 9, "proficiency_level": "Expert"},
            {"skill": "Analysis", "category": "Business",
             "years_of_experience": 6, "proficiency_level": "Expert"},
        ],
        "languages": [
            {"language": "Italian", "proficiency": "Intermediate", "rank": 2},
        ],
    }
