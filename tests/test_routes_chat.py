from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import create_app
from app.models import CandidateResult, ChatResponse, ParsedSpec


def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@h/d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))


def test_chat_returns_full_response(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    client = TestClient(create_app())

    fake_svc = MagicMock()
    fake_svc.search.return_value = ChatResponse(
        query="pharma",
        parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[CandidateResult(candidate_id="c1", rank=1, score=85,
                                   match_explanation="rag", highlights=["x"])],
        det_picks=[CandidateResult(candidate_id="c2", rank=1, score=0.82,
                                   per_dim={"industry": 1.0},
                                   match_explanation="det", highlights=["y"])],
        suggested=[CandidateResult(candidate_id="c1", rank=1, score=0.0,
                                   match_explanation="judge", highlights=["z"])],
        reasoning="agreement on c1",
    )
    with patch("app.routes.chat.get_search_service", return_value=fake_svc):
        r = client.post("/chat", json={"query": "pharma"})

    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "pharma"
    assert len(body["rag_picks"]) == 1
    assert body["rag_picks"][0]["candidate_id"] == "c1"
    assert len(body["det_picks"]) == 1
    assert body["det_picks"][0]["per_dim"] == {"industry": 1.0}
    assert len(body["suggested"]) == 1
    assert body["reasoning"] == "agreement on c1"
    fake_svc.search.assert_called_once_with("pharma")


def test_chat_accepts_conversation_id_noop(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    client = TestClient(create_app())
    fake_svc = MagicMock()
    fake_svc.search.return_value = ChatResponse(
        query="q", parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[], det_picks=[], suggested=[], reasoning="",
    )
    with patch("app.routes.chat.get_search_service", return_value=fake_svc):
        r = client.post("/chat", json={"query": "q", "conversation_id": "c-123"})
    assert r.status_code == 200
    # conversation_id is schema-valid but not yet used — search still runs
    fake_svc.search.assert_called_once_with("q")
