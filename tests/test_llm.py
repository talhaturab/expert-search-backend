from unittest.mock import MagicMock, patch

from app.llm import LLMClient


def test_chat_returns_text_content():
    mock = MagicMock()
    mock.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="hello"))]
    )
    with patch("app.llm.OpenAI", return_value=mock):
        c = LLMClient(api_key="x", model="any")
        out = c.chat([{"role": "user", "content": "hi"}])
    assert out == "hello"


def test_chat_json_parses_object():
    mock = MagicMock()
    mock.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"a": 1, "b": "x"}'))]
    )
    with patch("app.llm.OpenAI", return_value=mock):
        c = LLMClient(api_key="x", model="any")
        obj = c.chat_json([{"role": "user", "content": "return json"}])
    assert obj == {"a": 1, "b": "x"}


def test_chat_json_handles_markdown_fence():
    mock = MagicMock()
    mock.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='```json\n{"ok": true}\n```'))]
    )
    with patch("app.llm.OpenAI", return_value=mock):
        c = LLMClient(api_key="x", model="any")
        assert c.chat_json([{"role": "user", "content": "x"}]) == {"ok": True}
