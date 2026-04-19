"""rank_bm25 wrapper with simple whitespace + lowercase tokenization."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi
import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25Index:
    def __init__(self, bm25: BM25Okapi):
        self._bm25 = bm25

    @classmethod
    def build(cls, documents: list[str]) -> "BM25Index":
        tokenized = [tokenize(d) for d in documents]
        # rank_bm25 crashes on empty corpora; guard
        if not tokenized:
            tokenized = [[""]]
        return cls(BM25Okapi(tokenized))

    def score(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
