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
