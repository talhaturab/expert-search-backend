import time
from unittest.mock import patch

from app.models import ChatResponse, ParsedSpec
from app.session_store import SessionStore, SessionTurn


def _empty_response(query: str = "q") -> ChatResponse:
    return ChatResponse(
        query=query,
        conversation_id="placeholder",
        parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[], det_picks=[], suggested=[],
        reasoning="",
    )


def test_put_then_get_roundtrip():
    store = SessionStore(ttl_seconds=60)
    turn = SessionTurn(query="first", response=_empty_response(), timestamp=time.monotonic())
    store.put("cid-1", turn)
    got = store.get("cid-1")
    assert got is not None
    assert got.query == "first"


def test_get_returns_none_for_unknown_id():
    store = SessionStore(ttl_seconds=60)
    assert store.get("never-seen") is None


def test_entries_expire_after_ttl():
    store = SessionStore(ttl_seconds=60)
    old = SessionTurn(query="old", response=_empty_response(),
                     timestamp=time.monotonic() - 61)
    store.put("cid-old", old)
    assert store.get("cid-old") is None


def test_put_overwrites_previous_turn():
    store = SessionStore(ttl_seconds=60)
    t0 = time.monotonic()
    store.put("cid-1", SessionTurn(query="first",  response=_empty_response(), timestamp=t0))
    store.put("cid-1", SessionTurn(query="second", response=_empty_response(), timestamp=t0 + 1))
    got = store.get("cid-1")
    assert got is not None
    assert got.query == "second"
