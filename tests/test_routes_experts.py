from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import create_app


def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@h/d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))


def test_get_expert_returns_markdown(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    client = TestClient(create_app())
    with patch("app.routes.experts.render_profile_for_id",
               return_value="# Sara Ali\n..."):
        r = client.get("/experts/abc-123")
    assert r.status_code == 200
    body = r.json()
    assert body["candidate_id"] == "abc-123"
    assert "Sara Ali" in body["markdown"]


def test_get_expert_404_when_not_found(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    client = TestClient(create_app())
    with patch("app.routes.experts.render_profile_for_id",
               side_effect=LookupError("candidate not found: xyz")):
        r = client.get("/experts/xyz")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()
