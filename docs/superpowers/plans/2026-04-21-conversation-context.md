# Conversation Context — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add narrow-refinement follow-up support to `POST /chat` via `conversation_id`, so that *"Filter those to only people based in Saudi Arabia"* operates on the prior turn's `suggested` candidates instead of triggering a fresh full search.

**Architecture:** (1) In-memory `SessionStore` with 30-min TTL retains the last `ChatResponse` per `conversation_id`. (2) The query parser is extended to accept prior context and decide `is_refinement: bool`. (3) `SearchService` branches on the refinement flag: refinement path fetches the prior `suggested` candidates' bundles, skips the heavy full-scan retrieval, and runs `filter_and_score` + listwise rerank + judge directly on the ≤ 5 bundles. Fresh path is unchanged.

**Tech Stack:** Python 3.10+, Pydantic v2, FastAPI, pytest.

**Reference spec:** [`docs/superpowers/specs/2026-04-21-conversation-context-design.md`](../specs/2026-04-21-conversation-context-design.md)

---

## File plan

```
app/
├── session_store.py              # NEW — in-memory {conversation_id → last SessionTurn}
├── models.py                     # MODIFY — add PriorContext dataclass,
│                                   is_refinement to ParsedSpec,
│                                   conversation_id + is_refinement to ChatResponse
├── query_parser.py               # MODIFY — accept prior_context kwarg, append prior block to system prompt
├── search.py                     # MODIFY — inject SessionStore, branch on is_refinement
└── routes/chat.py                # MODIFY — wire SessionStore into get_search_service

tests/
├── test_session_store.py         # NEW
├── test_query_parser.py          # MODIFY — add two tests for prior-context behavior
├── test_search.py                # MODIFY — add refinement-path test
└── test_routes_chat.py           # MODIFY — add "conversation_id echoed, 2nd call is refinement" test
```

---

## Task 1 — `SessionStore` (in-memory session state with TTL)

**Files:**
- Create: `app/session_store.py`
- Test: `tests/test_session_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_session_store.py`:

```python
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
    # Stamp the turn as if it were written 61s ago
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
```

- [ ] **Step 2: Run the tests — expect failure**

```bash
poetry run pytest tests/test_session_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.session_store'`.

- [ ] **Step 3: Implement `app/session_store.py`**

```python
"""In-memory session store for conversational context.

Holds the last SessionTurn per conversation_id. Entries expire after `ttl_seconds`.
Eviction is lazy (performed on every get/put). Thread-safety via a single Lock —
good enough for the single-worker uvicorn default; swap for Redis if we ever
scale horizontally.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.models import ChatResponse


@dataclass
class SessionTurn:
    query: str
    response: ChatResponse
    timestamp: float  # time.monotonic() at write time


class SessionStore:
    def __init__(self, ttl_seconds: int = 1800):
        self._ttl = ttl_seconds
        self._data: dict[str, SessionTurn] = {}
        self._lock = threading.Lock()

    def get(self, conversation_id: str) -> SessionTurn | None:
        with self._lock:
            self._evict_expired_locked()
            return self._data.get(conversation_id)

    def put(self, conversation_id: str, turn: SessionTurn) -> None:
        with self._lock:
            self._evict_expired_locked()
            self._data[conversation_id] = turn

    def _evict_expired_locked(self) -> None:
        now = time.monotonic()
        stale = [cid for cid, t in self._data.items() if now - t.timestamp > self._ttl]
        for cid in stale:
            del self._data[cid]
```

- [ ] **Step 4: Run the tests — expect pass**

```bash
poetry run pytest tests/test_session_store.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/session_store.py tests/test_session_store.py
git commit -m "feat(session): in-memory SessionStore with TTL expiry"
```

---

## Task 2 — Models: `PriorContext` + `is_refinement` fields

**Files:**
- Modify: `app/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
def test_parsed_spec_has_is_refinement_default_false():
    from app.models import ParsedSpec
    spec = ParsedSpec(temporality="any")
    assert spec.is_refinement is False


def test_chat_response_carries_conversation_id_and_is_refinement():
    from app.models import ChatResponse, ParsedSpec
    resp = ChatResponse(
        query="q", conversation_id="abc-123", is_refinement=True,
        parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[], det_picks=[], suggested=[], reasoning="",
    )
    assert resp.conversation_id == "abc-123"
    assert resp.is_refinement is True


def test_chat_response_is_refinement_defaults_false():
    from app.models import ChatResponse, ParsedSpec
    resp = ChatResponse(
        query="q", conversation_id="x",
        parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[], det_picks=[], suggested=[], reasoning="",
    )
    assert resp.is_refinement is False


def test_prior_context_dataclass_exists_with_expected_shape():
    from app.models import PriorContext, ParsedSpec
    pc = PriorContext(
        prior_query="foo",
        prior_parsed_spec=ParsedSpec(temporality="any"),
        prior_suggested_ids=["a", "b"],
    )
    assert pc.prior_query == "foo"
    assert pc.prior_suggested_ids == ["a", "b"]
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_models.py -v -k "is_refinement or conversation_id or prior_context"
```

Expected: 4 failures — `is_refinement`, `conversation_id`, `PriorContext` don't exist yet.

- [ ] **Step 3: Modify `app/models.py`**

In `ParsedSpec`, add `is_refinement`:

```python
class ParsedSpec(BaseModel):
    function: DimensionSpec | None = None
    industry: DimensionSpec | None = None
    geography: GeoSpec | None = None
    seniority: SenioritySpec | None = None
    skills: SkillsSpec | None = None
    languages: LanguagesSpec | None = None
    min_years_exp: int | None = None
    temporality: Literal["current", "past", "any"] = "any"
    view_weights: ViewWeights | None = None
    is_refinement: bool = False   # NEW — True iff query refines a prior turn
```

In `ChatResponse`, add both new fields:

```python
class ChatResponse(BaseModel):
    query: str
    conversation_id: str                          # NEW — echoed so client can reuse
    is_refinement: bool = False                   # NEW
    parsed_spec: ParsedSpec
    rag_picks: list[CandidateResult]
    det_picks: list[CandidateResult]
    suggested: list[CandidateResult]
    reasoning: str
```

At the bottom of `app/models.py`, add the new dataclass (outside the Pydantic models — it's passed internally, never serialized):

```python
from dataclasses import dataclass


@dataclass
class PriorContext:
    """Internal container passed from SearchService into the query parser so it
    can decide whether the new turn is a refinement of the prior turn."""
    prior_query: str
    prior_parsed_spec: ParsedSpec
    prior_suggested_ids: list[str]
```

- [ ] **Step 4: Run — expect pass**

```bash
poetry run pytest tests/test_models.py -v
```

Expected: all tests pass (previously-passing tests still pass; 4 new tests now pass).

- [ ] **Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat(models): PriorContext + is_refinement on ParsedSpec + conversation_id on ChatResponse"
```

---

## Task 3 — Parser accepts prior context and classifies refinement

**Files:**
- Modify: `app/query_parser.py`
- Test: `tests/test_query_parser.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_query_parser.py`:

```python
from app.models import PriorContext


def test_parser_without_prior_context_does_not_mention_refinement_block():
    from unittest.mock import MagicMock
    llm = MagicMock()
    llm.chat_structured.return_value = ParsedSpec(temporality="any")
    parse_query("find pharma experts", llm=llm)
    system_msg = llm.chat_structured.call_args.kwargs["messages"][0]["content"]
    assert "PRIOR TURN CONTEXT" not in system_msg
    assert "REFINEMENT DETECTION" not in system_msg


def test_parser_with_prior_context_injects_block_with_ids_and_prior_query():
    from unittest.mock import MagicMock
    llm = MagicMock()
    llm.chat_structured.return_value = ParsedSpec(temporality="any", is_refinement=True)
    prior = PriorContext(
        prior_query="regulatory affairs in pharma in Middle East",
        prior_parsed_spec=ParsedSpec(temporality="any"),
        prior_suggested_ids=["abc-123", "def-456"],
    )
    spec = parse_query("filter those to only people in Saudi Arabia", llm=llm, prior_context=prior)
    assert spec.is_refinement is True
    system_msg = llm.chat_structured.call_args.kwargs["messages"][0]["content"]
    assert "PRIOR TURN CONTEXT" in system_msg
    assert "REFINEMENT DETECTION" in system_msg
    assert "regulatory affairs in pharma in Middle East" in system_msg
    assert "abc-123" in system_msg
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_query_parser.py -v -k "prior_context"
```

Expected: 2 failures — `parse_query` doesn't accept `prior_context`.

- [ ] **Step 3: Modify `app/query_parser.py`**

Update the imports block:

```python
from app.llm import LLMClient
from app.models import ParsedSpec, PriorContext
from app.vocabulary import Vocabulary
```

Add a helper to render the prior-context block:

```python
def _render_prior_context_block(prior: PriorContext) -> str:
    return (
        "\n\nPRIOR TURN CONTEXT:\n"
        f"- prior query: {prior.prior_query!r}\n"
        f"- prior result candidate IDs: {prior.prior_suggested_ids}\n"
        f"- prior parsed spec: {prior.prior_parsed_spec.model_dump(exclude_none=True)}\n"
        "\nREFINEMENT DETECTION:\n"
        "Set `is_refinement: true` ONLY if the new query narrows or filters the\n"
        "prior result set (phrases like 'filter those to ...', 'among them, only ...',\n"
        "'narrow to ...', 'from those, only ...'). A brand-new search that happens to\n"
        "share a topic is NOT a refinement.\n"
        "\n"
        "If `is_refinement: true`, produce a ParsedSpec containing ONLY the NEW\n"
        "constraints the user is adding. Do NOT re-emit the prior spec's constraints —\n"
        "the search will restrict to the prior candidate pool."
    )
```

Update `parse_query` signature and system-prompt assembly:

```python
def parse_query(
    query: str,
    llm: LLMClient,
    vocab: Vocabulary | None = None,
    prior_context: PriorContext | None = None,
) -> ParsedSpec:
    system = SYSTEM_PROMPT
    if vocab is not None:
        system = system + "\n\n" + vocab.to_prompt_block()
    if prior_context is not None:
        system = system + _render_prior_context_block(prior_context)

    spec = llm.chat_structured(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ],
        response_model=ParsedSpec,
        temperature=0.0,
        max_tokens=1200,
    )

    if vocab is not None:
        _restrict_to_vocabulary(spec, vocab)
    return spec
```

- [ ] **Step 4: Run — expect pass**

```bash
poetry run pytest tests/test_query_parser.py -v
```

Expected: all pass (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add app/query_parser.py tests/test_query_parser.py
git commit -m "feat(parser): optional prior_context enables refinement classification"
```

---

## Task 4 — `SearchService` branches on refinement

**Files:**
- Modify: `app/search.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`:

```python
import time

from app.models import ChatResponse, PriorContext
from app.session_store import SessionStore, SessionTurn


def test_search_generates_conversation_id_when_none_provided():
    svc = _make_service()
    resp = svc.search("find pharma experts")
    assert isinstance(resp.conversation_id, str) and len(resp.conversation_id) > 0


def test_search_echoes_provided_conversation_id_and_stores_turn():
    store = SessionStore(ttl_seconds=60)
    svc = _make_service(session_store=store)
    cid = "cid-fixed-123"
    resp = svc.search("first query", conversation_id=cid)
    assert resp.conversation_id == cid
    # SessionStore now has this turn
    assert store.get(cid) is not None


def test_refinement_path_restricts_pool_to_prior_suggested_ids():
    """When prior turn exists and parser flags is_refinement=True, search
    fetches bundles for prior suggested IDs only — not the full index."""
    fetched_ids: list[str] = []

    def fetch(cid: str) -> dict:
        fetched_ids.append(cid)
        return _fake_bundle(cid)

    # Prior turn to seed the session
    store = SessionStore(ttl_seconds=60)
    prior_response = ChatResponse(
        query="find pharma", conversation_id="cid-1",
        parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[], det_picks=[],
        suggested=[
            CandidateResult(candidate_id="c1", rank=1, score=0,
                            match_explanation="", highlights=[]),
            CandidateResult(candidate_id="c2", rank=2, score=0,
                            match_explanation="", highlights=[]),
        ],
        reasoning="",
    )
    store.put("cid-1",
              SessionTurn(query="find pharma", response=prior_response,
                          timestamp=time.monotonic()))

    # Parser returns is_refinement=True
    def parser_refine(q, prior_context=None):
        assert prior_context is not None
        assert prior_context.prior_suggested_ids == ["c1", "c2"]
        return ParsedSpec(temporality="any", is_refinement=True)

    # Deterministic called with only the restricted bundles
    captured_bundles: list[list[dict]] = []

    def det(bundles, spec, top_k=5):
        captured_bundles.append(bundles)
        return []

    svc = _make_service(
        session_store=store,
        parse_query=parser_refine,
        fetch_bundle=fetch,
        run_deterministic_on_pool=det,   # NEW injection point
    )
    svc.search("filter those to AE only", conversation_id="cid-1")

    # Only the prior IDs were fetched
    assert sorted(fetched_ids) == ["c1", "c2"]
    # Deterministic was run on exactly those 2 bundles
    assert len(captured_bundles) == 1
    assert len(captured_bundles[0]) == 2


def test_refinement_falls_back_to_fresh_search_when_prior_suggested_empty():
    """If prior turn's suggested list is empty, we can't narrow — run fresh."""
    store = SessionStore(ttl_seconds=60)
    store.put("cid-2", SessionTurn(
        query="q", response=ChatResponse(
            query="q", conversation_id="cid-2",
            parsed_spec=ParsedSpec(temporality="any"),
            rag_picks=[], det_picks=[], suggested=[],  # empty
            reasoning=""),
        timestamp=time.monotonic(),
    ))

    def parser(q, prior_context=None):
        return ParsedSpec(temporality="any", is_refinement=True)

    svc = _make_service(session_store=store, parse_query=parser)
    resp = svc.search("anything", conversation_id="cid-2")
    # Falls back to fresh — meaning is_refinement on the response is False
    assert resp.is_refinement is False
```

Also update the existing `_make_service` helper in `tests/test_search.py` to accept the two new kwargs (`session_store`, `run_deterministic_on_pool`) without breaking existing tests:

Find this function at the top of `tests/test_search.py`:

```python
def _make_service(**overrides) -> SearchService:
    # 1 candidate × 3 views × dim 4
    embs = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=np.float32)
    defaults = dict(
        parse_query=lambda q: ParsedSpec(temporality="any",
                                         view_weights=ViewWeights(summary=0.4, work=0.4, skills_edu=0.2)),
        embed_query=lambda t: [0.1] * 4,
        all_embeddings=embs,
        candidate_ids=["c1", "c1", "c1"],
        documents=["sum1", "work1", "skills1"],
        rag_rerank=_fake_rag,
        run_deterministic=_fake_det,
        judge=_fake_judge,
        fetch_bundle=_fake_bundle,
        rag_top_k=1,
    )
    defaults.update(overrides)
    return SearchService(**defaults)
```

Replace with:

```python
def _make_service(**overrides) -> SearchService:
    embs = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=np.float32)

    def _parse_with_optional_prior_kwarg(q, prior_context=None):
        return ParsedSpec(
            temporality="any",
            view_weights=ViewWeights(summary=0.4, work=0.4, skills_edu=0.2),
        )

    defaults = dict(
        parse_query=_parse_with_optional_prior_kwarg,
        embed_query=lambda t: [0.1] * 4,
        all_embeddings=embs,
        candidate_ids=["c1", "c1", "c1"],
        documents=["sum1", "work1", "skills1"],
        rag_rerank=_fake_rag,
        run_deterministic=_fake_det,
        run_deterministic_on_pool=lambda bundles, spec, top_k=5: _fake_det(spec),
        judge=_fake_judge,
        fetch_bundle=_fake_bundle,
        rag_top_k=1,
        session_store=SessionStore(ttl_seconds=60),
    )
    defaults.update(overrides)
    return SearchService(**defaults)
```

Add these imports at the top of `tests/test_search.py` if missing:

```python
from app.session_store import SessionStore, SessionTurn
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_search.py -v
```

Expected: existing tests break because `SearchService` doesn't accept `session_store` / `run_deterministic_on_pool`; the 4 new tests also fail.

- [ ] **Step 3: Modify `app/search.py`**

Replace the entire file with:

```python
"""End-to-end orchestrator. Ties parser + RAG + deterministic + judge into one call.

Dependency-injected — every stage is a callable passed in via the constructor.
Refinement: if the caller supplies a `conversation_id` whose last turn exists
and the parser classifies the new query as `is_refinement=True`, we restrict
the search to the prior turn's suggested candidates instead of running a fresh
full-scan retrieval.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.models import (
    CandidateResult, ChatResponse, ParsedSpec, PriorContext,
)
from app.profile_builder import render_mini
from app.rag_agent import retrieve_candidates
from app.session_store import SessionStore, SessionTurn


def _new_conversation_id() -> str:
    return uuid.uuid4().hex


@dataclass
class SearchService:
    # --- stage callables (injected) ---
    parse_query:       Callable[..., ParsedSpec]  # (query, *, prior_context=None) -> ParsedSpec
    embed_query:       Callable[[str], list[float]]
    rag_rerank:        Callable[[str, list[dict]], list[CandidateResult]]
    run_deterministic: Callable[[ParsedSpec], list[CandidateResult]]
    run_deterministic_on_pool: Callable[[list[dict], ParsedSpec, int], list[CandidateResult]]
    judge:             Callable[
        [str, list[CandidateResult], list[CandidateResult], dict[str, str]],
        tuple[list[CandidateResult], str],
    ]
    fetch_bundle:      Callable[[str], dict]

    # --- data loaded at startup ---
    all_embeddings: np.ndarray
    candidate_ids:  list[str]
    documents:      list[str]

    # --- runtime knobs ---
    rag_top_k: int = 50

    # --- session state ---
    session_store: SessionStore = field(default_factory=lambda: SessionStore(ttl_seconds=1800))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(self, query: str, conversation_id: str | None = None) -> ChatResponse:
        cid = conversation_id or _new_conversation_id()

        prior_turn = self.session_store.get(cid)
        prior_ctx = None
        if prior_turn and prior_turn.response.suggested:
            prior_ctx = PriorContext(
                prior_query=prior_turn.query,
                prior_parsed_spec=prior_turn.response.parsed_spec,
                prior_suggested_ids=[c.candidate_id for c in prior_turn.response.suggested],
            )

        spec = self.parse_query(query, prior_context=prior_ctx)

        is_refinement = (
            prior_ctx is not None
            and spec.is_refinement
            and len(prior_ctx.prior_suggested_ids) > 0
        )

        if is_refinement:
            response = self._refined_search(query, spec, prior_ctx)
        else:
            response = self._fresh_search(query, spec)

        response.conversation_id = cid
        response.is_refinement = is_refinement
        self.session_store.put(
            cid,
            SessionTurn(query=query, response=response, timestamp=time.monotonic()),
        )
        return response

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _fresh_search(self, query: str, spec: ParsedSpec) -> ChatResponse:
        q_vec = np.asarray(self.embed_query(query), dtype=np.float32)
        pool = retrieve_candidates(
            all_embs=self.all_embeddings,
            candidate_ids=self.candidate_ids,
            documents=self.documents,
            query_vec=q_vec,
            query_text=query,
            view_weights=spec.view_weights,
            top_k=self.rag_top_k,
        )
        rag_picks = self.rag_rerank(query, pool)
        det_picks = self.run_deterministic(spec)

        return self._finalize(query, spec, rag_picks, det_picks)

    def _refined_search(
        self, query: str, spec: ParsedSpec, prior_ctx: PriorContext
    ) -> ChatResponse:
        # Fetch bundles for just the prior suggested candidates.
        pool_bundles: list[dict] = []
        for cid_ in prior_ctx.prior_suggested_ids:
            try:
                pool_bundles.append(self.fetch_bundle(cid_))
            except Exception:
                pass

        # Deterministic scoring over the small pool.
        det_picks = self.run_deterministic_on_pool(pool_bundles, spec, 5)

        # RAG rerank straight on the pool (no retrieval needed — it's ≤ 5 docs).
        rag_input = [
            {
                "candidate_id": str(b["candidate"]["id"]),
                "score": 0.0,
                "rank": i + 1,
                "documents": {
                    "summary":    render_mini(b),
                    "work":       "",
                    "skills_edu": "",
                },
            }
            for i, b in enumerate(pool_bundles)
        ]
        rag_picks = self.rag_rerank(query, rag_input) if rag_input else []

        response = self._finalize(query, spec, rag_picks, det_picks)

        if not response.suggested:
            response.reasoning = (
                "No prior candidates matched the new constraint — "
                "try a broader follow-up or a fresh query."
            )
        return response

    # ------------------------------------------------------------------
    # Shared final stage: judge + ChatResponse assembly
    # ------------------------------------------------------------------
    def _finalize(
        self,
        query: str,
        spec: ParsedSpec,
        rag_picks: list[CandidateResult],
        det_picks: list[CandidateResult],
    ) -> ChatResponse:
        union_ids = (
            {r.candidate_id for r in rag_picks}
            | {r.candidate_id for r in det_picks}
        )
        profiles: dict[str, str] = {}
        for cid in union_ids:
            try:
                profiles[cid] = render_mini(self.fetch_bundle(cid))
            except Exception:
                profiles[cid] = "(profile unavailable)"

        if rag_picks or det_picks:
            suggested, reasoning = self.judge(query, rag_picks, det_picks, profiles)
        else:
            suggested, reasoning = [], ""

        return ChatResponse(
            query=query,
            conversation_id="",   # overwritten by caller
            is_refinement=False,  # overwritten by caller
            parsed_spec=spec,
            rag_picks=rag_picks,
            det_picks=det_picks,
            suggested=suggested,
            reasoning=reasoning,
        )
```

- [ ] **Step 4: Run — expect pass**

```bash
poetry run pytest tests/test_search.py -v
```

Expected: all search tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/search.py tests/test_search.py
git commit -m "feat(search): SearchService branches on refinement vs fresh search"
```

---

## Task 5 — Wire `SessionStore` into the `/chat` route factory

**Files:**
- Modify: `app/routes/chat.py`
- Test: `tests/test_routes_chat.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_routes_chat.py`:

```python
def test_chat_response_echoes_conversation_id_and_marks_refinement(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    client = TestClient(create_app())
    fake_svc = MagicMock()
    fake_svc.search.return_value = ChatResponse(
        query="q", conversation_id="cid-xyz", is_refinement=True,
        parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[], det_picks=[], suggested=[], reasoning="",
    )
    with patch("app.routes.chat.get_search_service", return_value=fake_svc):
        r = client.post("/chat", json={"query": "q", "conversation_id": "cid-xyz"})
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == "cid-xyz"
    assert body["is_refinement"] is True
    # Confirm conversation_id was threaded through to search()
    fake_svc.search.assert_called_once_with("q", conversation_id="cid-xyz")
```

Update the existing test (in the same file) that expects `search("pharma")` — it should now expect `search("pharma", conversation_id=None)`:

```python
def test_chat_returns_full_response(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    client = TestClient(create_app())

    fake_svc = MagicMock()
    fake_svc.search.return_value = ChatResponse(
        query="pharma", conversation_id="auto-generated-cid",
        parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[CandidateResult(candidate_id="c1", rank=1, score=85,
                                   match_explanation="rag", highlights=["x"])],
        det_picks=[CandidateResult(candidate_id="c2", rank=1, score=0.82,
                                   per_dim={"industry": 1.0},
                                   match_explanation="det", highlights=["y"])],
        suggested=[CandidateResult(candidate_id="c1", rank=1, score=0.0,
                                   match_explanation="judge", highlights=["z"])],
        reasoning="agreement on c1",
    )
    with patch("app.routes.chat.get_search_service", return_value=fake_svc):
        r = client.post("/chat", json={"query": "pharma"})

    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "pharma"
    assert len(body["rag_picks"]) == 1
    assert body["rag_picks"][0]["candidate_id"] == "c1"
    assert len(body["det_picks"]) == 1
    assert body["det_picks"][0]["per_dim"] == {"industry": 1.0}
    assert len(body["suggested"]) == 1
    assert body["reasoning"] == "agreement on c1"
    assert body["conversation_id"] == "auto-generated-cid"
    fake_svc.search.assert_called_once_with("pharma", conversation_id=None)
```

Also update the existing `test_chat_accepts_conversation_id_noop` to match the new call shape:

```python
def test_chat_threads_conversation_id_through_to_search(monkeypatch, tmp_path):
    _base_env(monkeypatch, tmp_path)
    client = TestClient(create_app())
    fake_svc = MagicMock()
    fake_svc.search.return_value = ChatResponse(
        query="q", conversation_id="c-123",
        parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[], det_picks=[], suggested=[], reasoning="",
    )
    with patch("app.routes.chat.get_search_service", return_value=fake_svc):
        r = client.post("/chat", json={"query": "q", "conversation_id": "c-123"})
    assert r.status_code == 200
    fake_svc.search.assert_called_once_with("q", conversation_id="c-123")
```

(Delete the old `test_chat_accepts_conversation_id_noop` — replaced by the two tests above.)

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_routes_chat.py -v
```

Expected: failures — `/chat` doesn't thread `conversation_id` through to `service.search(...)`, and `SearchService` constructor now requires `session_store` + `run_deterministic_on_pool`.

- [ ] **Step 3: Modify `app/routes/chat.py`**

Update the imports block to add `SessionStore` and `partial`:

```python
from app.session_store import SessionStore
```

Inside `get_search_service`, add the session store + pool-scoped deterministic wrapper, and include them in the `SearchService(...)` call:

```python
    # ... existing Chroma + bundles + vocab + clients loading ...

    session_store = SessionStore(ttl_seconds=1800)

    def _fetch_bundle(cid: str) -> dict:
        b = bundles_by_id.get(cid)
        if b is not None:
            return b
        return fetch_candidate_bundle(s.database_url, cid)

    def _det_on_pool(pool_bundles: list[dict], spec, top_k: int = 5):
        return filter_and_score(pool_bundles, spec, top_k=top_k)

    return SearchService(
        parse_query=lambda q, prior_context=None: parse_query(
            q, llm=llm, vocab=vocab, prior_context=prior_context
        ),
        embed_query=lambda t: embedder.embed_one(t),
        all_embeddings=embs,
        candidate_ids=candidate_ids_in_order,
        documents=docs,
        rag_rerank=lambda q, pool: rerank_and_explain(
            q, pool, llm=llm, top_k=s.final_top_k
        ),
        run_deterministic=lambda spec: filter_and_score(
            all_bundles, spec, top_k=s.deterministic_top_k
        ),
        run_deterministic_on_pool=_det_on_pool,
        judge=lambda q, rp, dp, pm: cherry_pick_top_five(q, rp, dp, pm, llm=llm),
        fetch_bundle=_fetch_bundle,
        rag_top_k=s.rag_top_k,
        session_store=session_store,
    )
```

Update the `chat` handler to thread `conversation_id`:

```python
@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    service = get_search_service()
    return service.search(req.query, conversation_id=req.conversation_id)
```

- [ ] **Step 4: Run — expect pass**

```bash
poetry run pytest tests/test_routes_chat.py -v
```

Expected: all pass.

- [ ] **Step 5: Full unit suite sanity**

```bash
poetry run pytest -m "not integration" -q
```

Expected: all pass (previous + new tests).

- [ ] **Step 6: Commit**

```bash
git add app/routes/chat.py tests/test_routes_chat.py
git commit -m "feat(api): thread conversation_id through /chat and wire SessionStore"
```

---

## Task 6 — README note + live smoke test

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a subsection under "Out of scope for this iteration"**

Replace the existing "Conversational context" bullet with a new "Conversational context" subsection **above** Out of scope:

```markdown
## Conversational context

Follow-up queries can reference prior results via `conversation_id`:

```bash
# First turn — server returns a conversation_id in the response
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "regulatory affairs experts in pharma in the Middle East"}' | jq '{conversation_id, suggested}'

# Second turn — reuse the conversation_id to narrow the prior result set
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "filter those to only people in Saudi Arabia", "conversation_id": "<paste from above>"}' | jq '{is_refinement, suggested}'
```

The second call takes the prior turn's `suggested` candidates (up to 5) and
re-scores them against the new constraint instead of running a fresh full-scan
search. Sessions live 30 minutes in-memory and are lost on process restart.

Only *narrowing* follow-ups are supported in this iteration (e.g., *"filter
those to..."*, *"among them, only..."*). Broadening follow-ups or explanation
questions are not.
```

And delete the now-stale "Conversational context" bullet under "Out of scope for this iteration".

- [ ] **Step 2: Live smoke test**

```bash
rm -f /tmp/chat1.json /tmp/chat2.json
poetry run uvicorn app.main:app --port 8000 > /tmp/uvicorn.log 2>&1 &
UVI=$!
# Wait for ready
until curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health 2>/dev/null | grep -q 200; do sleep 1; done
```

Expected: server up.

```bash
# Turn 1
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "regulatory affairs experts in pharma in the Middle East"}' > /tmp/chat1.json
CID=$(python3 -c "import json; print(json.load(open('/tmp/chat1.json'))['conversation_id'])")
echo "cid=$CID"
python3 -c "import json; r=json.load(open('/tmp/chat1.json')); print('turn1 suggested:', [c['candidate_id'][:8] for c in r['suggested']])"
```

Expected: a non-empty conversation_id + 5 suggested candidate IDs.

```bash
# Turn 2 — narrow
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"filter those to only people based in Saudi Arabia\", \"conversation_id\": \"$CID\"}" > /tmp/chat2.json
python3 -c "
import json
r = json.load(open('/tmp/chat2.json'))
print('is_refinement:', r['is_refinement'])
print('suggested:', [c['candidate_id'][:8] for c in r['suggested']])
print('reasoning:', r['reasoning'][:120])
"
```

Expected: `is_refinement: True` and either a subset of turn 1's suggested
candidates or the empty-pool reasoning string.

```bash
kill $UVI
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): document conversation_id + narrow-refinement follow-ups"
```

---

## Self-review against the spec

- ✅ **§3 Flow** — Tasks 4 + 5 implement both paths (fresh + refined) and the SessionStore round-trip.
- ✅ **§4.1 SessionStore** — Task 1 covers interface + TTL + overwrite semantics with a 4-test suite.
- ✅ **§4.2 Parser extension** — Task 3 covers the `prior_context` kwarg, prompt block, and the system-message assertions.
- ✅ **§4.3 SearchService** — Task 4 has tests for conversation_id generation, echo, refinement-restricts-pool, and the "empty suggested → fallback to fresh" case.
- ✅ **§4.5 API shape** — Task 2 (models) + Task 5 (/chat route) cover the new fields.
- ✅ **§5 Storage/TTL** — 30 min default (SESSION_TTL_SECONDS knob left for future env override; default is baked into SessionStore and SearchService).
- ✅ **§6 Testing plan** — all 5 testing bullets have tasks; live end-to-end is Task 6.
- ✅ **§9 Cross-check** — satisfied by Tasks 4 + 6 (smoke test runs the brief's exact example).

No placeholders, no "TBD"s, every task has complete code. Types are consistent across tasks: `PriorContext`, `SessionTurn`, `SessionStore`, `is_refinement`, `conversation_id` all defined in Tasks 1/2 and referenced with identical signatures in Tasks 3/4/5.

---
