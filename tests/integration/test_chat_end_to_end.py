"""End-to-end smoke test — hits real Postgres + real OpenRouter + real Chroma.

Skipped automatically unless OPENROUTER_API_KEY is set AND the Chroma index
has already been populated via POST /ingest (or the CLI equivalent).

Run explicitly:
    poetry run pytest tests/integration -v -m integration
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient


# Load the repo's .env so integration tests can find OPENROUTER_API_KEY etc.
# (Unit tests deliberately isolate from .env; integration tests need it.)
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from app.main import create_app  # noqa: E402 — must follow env loading


pytestmark = pytest.mark.integration


def _env_or_skip():
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")
    # Also skip if the Chroma persist dir is missing / empty.
    chroma_dir = Path(os.environ.get("CHROMA_PERSIST_PATH", "./data/chroma"))
    if not chroma_dir.exists() or not (chroma_dir / "chroma.sqlite3").exists():
        pytest.skip(f"Chroma not populated at {chroma_dir}. Run POST /ingest first.")


def test_health_reports_ok():
    _env_or_skip()
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["checks"]["chroma_populated"] is True


def test_chat_end_to_end_returns_full_response():
    _env_or_skip()
    client = TestClient(create_app())
    r = client.post("/chat", json={
        "query": "regulatory affairs experts in pharma in the Middle East",
    })
    assert r.status_code == 200
    body = r.json()

    # Top-level shape per the spec
    assert body["query"].startswith("regulatory")
    assert "parsed_spec" in body
    assert isinstance(body["rag_picks"], list)
    assert isinstance(body["det_picks"], list)
    assert isinstance(body["suggested"], list)
    assert isinstance(body["reasoning"], str)

    # Every returned expert must carry the four brief-required fields
    for key in ("rag_picks", "suggested"):
        for r_ in body[key]:
            assert "candidate_id" in r_
            assert "match_explanation" in r_
            assert "highlights" in r_
            assert "rank" in r_
    # Deterministic picks additionally carry per_dim breakdown when scored
    if body["det_picks"]:
        assert body["det_picks"][0].get("per_dim") is not None
