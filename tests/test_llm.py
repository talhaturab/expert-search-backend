from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from app.llm import LLMClient


class _Sample(BaseModel):
    city: str
    pop: int


def test_chat_returns_text_content():
    mock = MagicMock()
    mock.responses.create.return_value = MagicMock(output_text="hello")
    with patch("app.llm.OpenAI", return_value=mock):
        c = LLMClient(api_key="x", model="any")
        out = c.chat([{"role": "user", "content": "hi"}])
    assert out == "hello"
    # Confirm Responses API was used (not chat.completions)
    mock.responses.create.assert_called_once()


def test_chat_structured_returns_typed_pydantic_instance():
    parsed_instance = _Sample(city="Tokyo", pop=13960000)
    mock = MagicMock()
    mock.responses.parse.return_value = MagicMock(output_parsed=parsed_instance)
    with patch("app.llm.OpenAI", return_value=mock):
        c = LLMClient(api_key="x", model="any")
        result = c.chat_structured(
            messages=[{"role": "user", "content": "Tokyo?"}],
            response_model=_Sample,
        )
    assert isinstance(result, _Sample)
    assert result.city == "Tokyo"
    assert result.pop == 13960000
    # Confirm text_format was the Pydantic model
    call_kwargs = mock.responses.parse.call_args.kwargs
    assert call_kwargs["text_format"] is _Sample


def test_chat_structured_raises_when_parsed_is_none():
    mock = MagicMock()
    mock.responses.parse.return_value = MagicMock(
        output_parsed=None, output_text="gibberish"
    )
    with patch("app.llm.OpenAI", return_value=mock):
        c = LLMClient(api_key="x", model="any")
        try:
            c.chat_structured(
                messages=[{"role": "user", "content": "x"}],
                response_model=_Sample,
            )
        except ValueError as e:
            assert "no parsed output" in str(e)
        else:
            raise AssertionError("Expected ValueError when parsed is None")
