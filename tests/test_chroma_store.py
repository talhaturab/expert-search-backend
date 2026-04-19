import pytest

from app.chroma_store import ChromaStore


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_path=str(tmp_path / "chroma"))


def test_upsert_and_load_roundtrip(store):
    store.upsert_batch(
        ids=["c-1::summary", "c-1::work", "c-2::summary"],
        embeddings=[[0.1]*3, [0.2]*3, [0.3]*3],
        documents=["sum1", "work1", "sum2"],
        metadatas=[
            {"candidate_id": "c-1", "view": "summary"},
            {"candidate_id": "c-1", "view": "work"},
            {"candidate_id": "c-2", "view": "summary"},
        ],
    )
    embs, meta, docs, ids = store.load_all()
    assert embs.shape == (3, 3)
    assert len(ids) == 3
    assert all(m["view"] in {"summary", "work"} for m in meta)


def test_count_after_upsert(store):
    assert store.count() == 0
    store.upsert_batch(
        ids=["c-1::summary"],
        embeddings=[[0.5]*3],
        documents=["x"],
        metadatas=[{"candidate_id": "c-1", "view": "summary"}],
    )
    assert store.count() == 1
