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
        "candidates_loaded": 10, "documents_to_process": 30,
        "documents_written": 30, "documents_skipped_existing": 0,
        "stopped_reason": None, "duration_seconds": 1.2,
    }) as mock_build:
        r = client.post("/ingest", json={"force": True})

    assert r.status_code == 200
    body = r.json()
    assert body["candidates_loaded"] == 10
    assert body["documents_written"] == 30
    assert body["stopped_reason"] is None
    # settings.ingest_limit (200) should have been threaded into build_index
    kwargs = mock_build.call_args.kwargs
    assert kwargs["limit"] == 200
    assert kwargs["reset"] is False  # default


def test_ingest_threads_reset_flag_to_build_index(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    client = TestClient(create_app())
    with patch("app.routes.ingest.build_index", return_value={
        "candidates_loaded": 0, "documents_to_process": 0,
        "documents_written": 0, "documents_skipped_existing": 0,
        "stopped_reason": None, "duration_seconds": 0.0,
    }) as mock_build:
        r = client.post("/ingest", json={"reset": True})
    assert r.status_code == 200
    assert mock_build.call_args.kwargs["reset"] is True


def test_ingest_skips_when_already_populated_and_not_forced(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    client = TestClient(create_app())
    with patch("app.routes.ingest.ChromaStore") as MockStore:
        MockStore.return_value.count.return_value = 5
        with patch("app.routes.ingest.build_index") as mock_build:
            r = client.post("/ingest", json={"force": False})

    assert r.status_code == 200
    body = r.json()
    assert body["documents_written"] == 0
    assert body["stopped_reason"] == "already_populated"
    mock_build.assert_not_called()
