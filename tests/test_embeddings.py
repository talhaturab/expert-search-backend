from unittest.mock import MagicMock, patch

from app.embeddings import EmbeddingClient


def test_embed_single_returns_float_vector():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )
    with patch("app.embeddings.OpenAI", return_value=mock_client):
        c = EmbeddingClient(api_key="x", model="any/model")
        vec = c.embed_one("hello")
    assert vec == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once()


def test_embed_batch_returns_list_of_vectors():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1]*3), MagicMock(embedding=[0.2]*3)]
    )
    with patch("app.embeddings.OpenAI", return_value=mock_client):
        c = EmbeddingClient(api_key="x", model="any/model")
        vecs = c.embed_batch(["a", "b"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 3
