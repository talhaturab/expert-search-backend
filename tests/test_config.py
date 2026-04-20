import os
from app.config import Settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@host:5432/db")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", "/tmp/chroma")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

    s = Settings(_env_file=None)  # isolate from the repo's .env

    assert s.database_url.endswith("/db")
    assert s.openrouter_api_key == "sk-or-test"
    assert s.chroma_persist_path == "/tmp/chroma"
    assert s.hyde_enabled is True  # default
    assert s.rag_top_k == 50       # default


def test_settings_defaults_for_optional_fields(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")

    s = Settings(_env_file=None)  # isolate from the repo's .env

    assert s.hyde_enabled is True
    assert s.rag_top_k == 50
    assert s.deterministic_top_k == 5
    assert s.final_top_k == 5
