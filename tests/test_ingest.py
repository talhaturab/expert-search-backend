from unittest.mock import patch

from app.ingest import build_index
from app.chroma_store import ChromaStore


def test_build_index_forwards_limit_to_fetch_all_bundles(tmp_path, sample_bundle):
    """`limit` arg on build_index must be threaded into fetch_all_bundles."""
    store = ChromaStore(persist_path=str(tmp_path / "chroma"))
    with patch("app.ingest.fetch_all_bundles", return_value=[sample_bundle]) as mock_fetch:
        with patch("app.ingest.EmbeddingClient") as EmbClient:
            EmbClient.return_value.embed_batch.return_value = [[0.1, 0.2, 0.3]] * 3
            build_index(dsn="fake", api_key="x", embedding_model="m",
                        store=store, limit=5)
    mock_fetch.assert_called_once_with("fake", limit=5)


def test_build_index_upserts_three_docs_per_candidate(tmp_path, sample_bundle):
    store = ChromaStore(persist_path=str(tmp_path / "chroma"))
    with patch("app.ingest.fetch_all_bundles", return_value=[sample_bundle]):
        with patch("app.ingest.EmbeddingClient") as EmbClient:
            EmbClient.return_value.embed_batch.return_value = [[0.1, 0.2, 0.3]] * 3
            result = build_index(dsn="postgresql://fake", api_key="x",
                                 embedding_model="m", store=store, reset=True)

    assert result["candidates_loaded"] == 1
    assert result["documents_to_process"] == 3
    assert result["documents_written"] == 3
    assert result["documents_skipped_existing"] == 0
    assert result["stopped_reason"] is None
    assert store.count() == 3


def test_build_index_resume_skips_already_ingested(tmp_path, sample_bundle):
    """Running twice without reset should skip all 3 already-embedded docs."""
    store = ChromaStore(persist_path=str(tmp_path / "chroma"))
    with patch("app.ingest.fetch_all_bundles", return_value=[sample_bundle]):
        with patch("app.ingest.EmbeddingClient") as EmbClient:
            EmbClient.return_value.embed_batch.return_value = [[0.1, 0.2, 0.3]] * 3
            # First run: fresh, writes 3 docs
            build_index(dsn="x", api_key="x", embedding_model="m",
                        store=store, reset=True)
            # Second run: same bundle, no reset -> all 3 already in Chroma
            EmbClient.return_value.embed_batch.reset_mock()
            result = build_index(dsn="x", api_key="x", embedding_model="m",
                                 store=store, reset=False)
    assert result["documents_to_process"] == 0
    assert result["documents_written"] == 0
    assert result["documents_skipped_existing"] == 3
    EmbClient.return_value.embed_batch.assert_not_called()


def test_build_index_stops_on_rate_limit_and_reports_progress(tmp_path, sample_bundle):
    """RateLimitError during embed_batch -> stop cleanly, partial progress returned."""
    from openai import RateLimitError
    import httpx

    # Two bundles -> 6 docs; with batch_size=3, first batch succeeds, second fails.
    second_bundle = {**sample_bundle}
    second_bundle["candidate"] = {**sample_bundle["candidate"], "id": "cid-002"}

    store = ChromaStore(persist_path=str(tmp_path / "chroma"))
    with patch("app.ingest.fetch_all_bundles", return_value=[sample_bundle, second_bundle]):
        with patch("app.ingest.EmbeddingClient") as EmbClient:
            mock_emb = EmbClient.return_value
            # First call succeeds, second raises rate limit
            mock_response = httpx.Response(429, request=httpx.Request("POST", "http://x"))
            mock_emb.embed_batch.side_effect = [
                [[0.1, 0.2, 0.3]] * 3,  # first batch OK
                RateLimitError("rate limited", response=mock_response, body=None),
            ]
            result = build_index(dsn="x", api_key="x", embedding_model="m",
                                 store=store, batch_size=3, reset=True)

    assert result["documents_written"] == 3  # only the first batch made it
    assert result["stopped_reason"] is not None
    assert "rate_limit" in result["stopped_reason"]
