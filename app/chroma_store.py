"""Chroma wrapper. We use a single collection 'candidate_profiles'."""
from __future__ import annotations

from pathlib import Path

import chromadb
import numpy as np

COLLECTION_NAME = "candidate_profiles"


class ChromaStore:
    def __init__(self, persist_path: str):
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(name=COLLECTION_NAME)

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(name=COLLECTION_NAME)

    def upsert_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def load_all(self) -> tuple[np.ndarray, list[dict], list[str], list[str]]:
        """Load EVERYTHING into memory. Returns (embeddings_ndarray, metadatas, documents, ids)."""
        data = self._collection.get(include=["embeddings", "metadatas", "documents"])
        embs = np.asarray(data["embeddings"], dtype=np.float32)
        return embs, data["metadatas"], data["documents"], data["ids"]
