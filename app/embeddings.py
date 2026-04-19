"""OpenRouter-compatible embedding client (uses the OpenAI SDK, swapped base_url)."""
from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class EmbeddingClient:
    def __init__(self, api_key: str, model: str, base_url: str = OPENROUTER_BASE):
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def embed_one(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
