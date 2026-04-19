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
            mock_emb = EmbClient.return_value
            mock_emb.embed_batch.return_value = [[0.1, 0.2, 0.3]] * 3

            result = build_index(
                dsn="postgresql://fake",
                api_key="x",
                embedding_model="m",
                store=store,
            )

    assert result["candidates_indexed"] == 1
    assert result["documents_written"] == 3
    assert store.count() == 3

    embs, meta, docs, ids = store.load_all()
    views = {m["view"] for m in meta}
    assert views == {"summary", "work", "skills_edu"}
