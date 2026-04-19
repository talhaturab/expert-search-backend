from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import create_app


def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@h/d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("INGEST_LIMIT", "200")


def test_ingest_returns_summary(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    client = TestClient(create_app())

    with patch("app.routes.ingest.build_index", return_value={
        "candidates_indexed": 10, "documents_written": 30, "duration_seconds": 1.2,
    }) as mock_build:
        r = client.post("/ingest", json={"force": True})

    assert r.status_code == 200
    body = r.json()
    assert body["candidates_indexed"] == 10
    assert body["documents_written"] == 30
    # settings.ingest_limit (200) should have been threaded into build_index
    kwargs = mock_build.call_args.kwargs
    assert kwargs["limit"] == 200


def test_ingest_skips_when_already_populated_and_not_forced(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    client = TestClient(create_app())

    # Pretend chroma already has content
    with patch("app.routes.ingest.ChromaStore") as MockStore:
        MockStore.return_value.count.return_value = 5
        with patch("app.routes.ingest.build_index") as mock_build:
            r = client.post("/ingest", json={"force": False})

    assert r.status_code == 200
    body = r.json()
    assert body["candidates_indexed"] == 0
    assert body["documents_written"] == 0
    mock_build.assert_not_called()
