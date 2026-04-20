import pytest

from app.vocabulary import Vocabulary, load_vocabulary


def test_vocabulary_to_prompt_block_includes_all_four_fields():
    v = Vocabulary(
        industries=["Pharmaceuticals", "Finance"],
        skill_categories=["Engineering", "Marketing"],
        languages=["English", "Arabic"],
        proficiency_levels=["Beginner", "Intermediate", "Fluent", "Native"],
        skills=["python", "java"],
    )
    block = v.to_prompt_block()
    assert "INDUSTRIES" in block
    assert "Pharmaceuticals" in block
    assert "SKILL CATEGORIES" in block
    assert "Engineering" in block
    assert "LANGUAGES" in block
    assert "Arabic" in block
    assert "PROFICIENCY LEVELS" in block
    assert "Fluent" in block
    # `skills` is NOT in the prompt block (too big); used only for downstream fuzzy match
    assert "python" not in block
    assert "java" not in block


@pytest.mark.integration
def test_load_vocabulary_from_real_db_returns_populated_lists(real_db_settings):
    v = load_vocabulary(real_db_settings.database_url)
    # Sanity on counts (matches the earlier probe)
    assert len(v.industries) >= 200
    assert len(v.skill_categories) == 112
    assert len(v.languages) == 49
    assert len(v.proficiency_levels) == 4
    assert len(v.skills) == 1551
    # Lists are sorted / deduped
    assert "Pharmaceuticals" in v.industries
    assert "Native" in v.proficiency_levels
