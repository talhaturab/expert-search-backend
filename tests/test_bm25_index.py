from app.bm25_index import BM25Index


def test_bm25_scores_for_matching_query():
    docs = ["python expert regulatory affairs", "sales manager pharma", "pharma regulatory"]
    idx = BM25Index.build(docs)
    scores = idx.score("regulatory pharma")
    assert len(scores) == 3
    # doc 2 (pharma regulatory) should score highest
    assert scores[2] > scores[0]
    assert scores[2] > scores[1]


def test_bm25_tokenizer_lowercase():
    # Query case must not affect scores (tokenizer lowercases both sides).
    docs = ["Python EXPERT", "java developer", "python engineer"]
    idx = BM25Index.build(docs)
    lower = idx.score("python")
    upper = idx.score("PYTHON")
    assert (lower == upper).all()
    # Docs containing "python" should outrank the one that doesn't.
    assert lower[0] > lower[1]
    assert lower[2] > lower[1]
