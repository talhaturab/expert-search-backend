import pytest
from app.db import fetch_all_candidates, fetch_candidate_bundle


@pytest.mark.integration
def test_fetch_all_candidates_returns_10k(real_db_settings):
    rows = fetch_all_candidates(real_db_settings.database_url)
    assert len(rows) == 10120
    assert "id" in rows[0]
    assert "first_name" in rows[0]
    assert "city" in rows[0]           # enriched
    assert "nationality" in rows[0]    # enriched


@pytest.mark.integration
def test_fetch_candidate_bundle_for_known_candidate(real_db_settings):
    # Sara Ali (from exploration)
    cid = "70222c8e-2b7a-4a9e-bc42-9ae3eaa2a89a"
    bundle = fetch_candidate_bundle(real_db_settings.database_url, cid)

    assert bundle["candidate"]["first_name"] == "Sara"
    assert bundle["candidate"]["last_name"] == "Ali"
    assert len(bundle["work"]) >= 1
    assert len(bundle["education"]) >= 1
    assert len(bundle["skills"]) >= 1
