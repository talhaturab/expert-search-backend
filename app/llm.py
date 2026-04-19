"""OpenRouter-compatible LLM client (uses OpenAI SDK with base_url override)."""
from __future__ import annotations

import json
import re

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


OPENROUTER_BASE = "https://openrouter.ai/api/v1"


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class LLMClient:
    def __init__(self, api_key: str, model: str, base_url: str = OPENROUTER_BASE):
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_json(self, messages: list[dict], temperature: float = 0.0) -> dict:
        """Call chat and parse content as JSON; tolerate ```json fences."""
        raw = self.chat(messages, temperature=temperature)
        m = _FENCE_RE.match(raw)
        text = m.group(1) if m else raw
        return json.loads(text)
