import json
from unittest.mock import MagicMock, patch

from app.cli import main
from app.models import ChatResponse, ParsedSpec


def _base_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@h/d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))


def test_cli_chat_prints_json_response(monkeypatch, capsys, tmp_path):
    _base_env(monkeypatch, tmp_path)
    svc = MagicMock()
    svc.search.return_value = ChatResponse(
        query="q",
        parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[], det_picks=[], suggested=[],
        reasoning="no candidates",
    )
    with patch("app.cli.get_search_service", return_value=svc):
        rc = main(["chat", "pharma experts"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["query"] == "q"
    assert parsed["reasoning"] == "no candidates"


def test_cli_ingest_prints_summary(monkeypatch, capsys, tmp_path):
    _base_env(monkeypatch, tmp_path)
    with patch("app.cli.build_index",
               return_value={"candidates_indexed": 42, "documents_written": 126,
                             "duration_seconds": 5.0}):
        rc = main(["ingest", "--force"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "42" in out
    assert "126" in out


def test_cli_ingest_skips_when_populated_and_not_forced(monkeypatch, capsys, tmp_path):
    _base_env(monkeypatch, tmp_path)
    with patch("app.cli.ChromaStore") as MockStore:
        MockStore.return_value.count.return_value = 10
        with patch("app.cli.build_index") as mock_build:
            rc = main(["ingest"])  # no --force
    assert rc == 1
    mock_build.assert_not_called()
    err = capsys.readouterr().err
    assert "already populated" in err.lower()
