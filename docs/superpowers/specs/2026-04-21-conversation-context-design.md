# Conversation Context — Design

**Date:** 2026-04-21
**Status:** Approved design; awaiting implementation plan
**Scope:** Add narrow-refinement follow-up support to `POST /chat` via `conversation_id`

---

## 1. Goal

Satisfy the brief's requirement:

> *"Support conversational context through a session or conversation ID, so follow-up queries can reference prior results (e.g., 'Filter those to only people based in Saudi Arabia')."*

**In scope (Option A — narrow-only refinement):**

- Client-supplied or server-generated `conversation_id` maintains a session across turns.
- Follow-up queries that narrow the prior turn's results (e.g., *"filter those to X"*, *"among them, only Y"*) re-score the prior `suggested` candidates under the new constraint instead of running a fresh search.
- The parser decides "refinement vs. fresh query" given prior context.

**Out of scope** (not required by the brief; clean extension later):

- Broadening/similar queries (*"show me more like these"*).
- Explanation/QA turns (*"why did you pick #3?"*).
- Cross-session persistence (survives process restart).
- Multi-step reasoning beyond one level of refinement per turn (we always refine relative to the immediately-prior turn).

## 2. Non-goals

- Durable storage (Redis, Postgres, disk). In-memory is enough.
- Multi-user auth / access control on sessions. Anyone with the `conversation_id` can use it.
- Concurrency control beyond a process-level `threading.Lock`.
- Automatic disambiguation of "those" vs a specific candidate.

## 3. High-level flow

```
POST /chat {query, conversation_id?}
        │
        ▼
  SessionStore.get(conversation_id)
        │                        │
    prior turn exists?        no prior turn?
        │                        │
        ▼                        ▼
  parse_query(q, prior_ctx)   parse_query(q, None)
        │                        │
        ▼                        ▼
    is_refinement?            is_refinement = False
     │        │
     y        n
     │        │
     ▼        ▼
 run on   run full
 prior    pipeline
 IDs only
        │
        ▼
  Judge → ChatResponse (with conversation_id echoed)
        │
        ▼
  SessionStore.put(conversation_id, new turn)
```

## 4. Components

### 4.1 `SessionStore` — new module `app/session_store.py`

In-memory session state with TTL. Thread-safe via a single `Lock`.

```python
@dataclass
class SessionTurn:
    query: str
    response: ChatResponse           # full response so we can reference spec + suggested
    timestamp: float                 # monotonic, for TTL

class SessionStore:
    def __init__(self, ttl_seconds: int = 1800): ...
    def get(self, conversation_id: str) -> SessionTurn | None: ...
    def put(self, conversation_id: str, turn: SessionTurn) -> None: ...
    def _evict_expired(self) -> None: ...  # called lazily on get/put
```

- Storage: `dict[str, SessionTurn]` (only the last turn per session — Option A's narrow refinement operates relative to the immediately-prior turn, not the full history).
- Eviction: lazy — on every `get`/`put`, purge entries whose timestamp is > `ttl_seconds` old.
- Concurrency: one `threading.Lock` around all dict operations. Good enough for the FastAPI single-worker case; if we ever run multi-worker, we'd swap to Redis.

### 4.2 Parser extension — `app/query_parser.py`

The existing `parse_query(query, llm, vocab)` gains one optional parameter:

```python
def parse_query(
    query: str,
    llm: LLMClient,
    vocab: Vocabulary | None = None,
    prior_context: PriorContext | None = None,   # NEW
) -> ParsedSpec:
```

Where `PriorContext` is a small shape passed down from `SearchService`:

```python
@dataclass
class PriorContext:
    prior_query: str
    prior_parsed_spec: ParsedSpec
    prior_suggested_ids: list[str]   # the candidate_ids the user saw
```

**Prompt change:** when `prior_context` is supplied, append a block:

```
PRIOR TURN CONTEXT:
- prior query: "regulatory affairs in pharma in the Middle East"
- prior result candidate IDs: ["abc-...", "def-...", ...]  (5 IDs)
- prior parsed spec: {industry: Pharmaceuticals, geography: [AE, SA, ...], ...}

REFINEMENT DETECTION:
Set `is_refinement: true` ONLY if the new query narrows or filters the prior
result set (phrases like "filter those to...", "among them, only...",
"narrow to...", "from those, only..."). A brand-new search that happens to
share a topic is NOT a refinement.

If `is_refinement: true`, produce a ParsedSpec containing ONLY the new
constraints the user is adding — do not re-emit the prior spec's constraints,
the search will restrict to the prior candidate pool.
```

**Model change:** add `is_refinement: bool = False` to `ParsedSpec`. Default False when no context supplied.

### 4.3 `SearchService` extension

One new code path for the refinement case, plus `conversation_id` bookkeeping.

```python
def search(self, query: str, conversation_id: str | None = None) -> ChatResponse:
    # 1. Session lookup
    cid = conversation_id or _new_conversation_id()
    prior_turn = self.session_store.get(cid)  # returns None if missing/expired

    # 2. Parse (with prior context if any)
    prior_ctx = None
    if prior_turn:
        prior_ctx = PriorContext(
            prior_query=prior_turn.query,
            prior_parsed_spec=prior_turn.response.parsed_spec,
            prior_suggested_ids=[c.candidate_id for c in prior_turn.response.suggested],
        )
    spec = self.parse_query(query, prior_context=prior_ctx)

    # 3. Decide path
    is_refinement = bool(prior_ctx) and spec.is_refinement \
                    and len(prior_ctx.prior_suggested_ids) > 0

    if is_refinement:
        response = self._refined_search(query, spec, prior_ctx)
    else:
        response = self._fresh_search(query, spec)

    # 4. Store + echo conversation_id
    response.conversation_id = cid
    response.is_refinement = is_refinement
    self.session_store.put(cid, SessionTurn(query=query, response=response, timestamp=time.monotonic()))
    return response
```

**`_fresh_search`** is the existing pipeline — unchanged.

**`_refined_search`** — new path. The candidate pool is ≤ 5; we skip heavy retrieval:

```python
def _refined_search(self, query, spec, prior_ctx):
    # Fetch bundles for just the prior IDs
    pool_bundles = [self.fetch_bundle(cid) for cid in prior_ctx.prior_suggested_ids]

    # Deterministic: score over the tiny pool using the *new* spec only.
    det_picks = self.run_deterministic_on_pool(pool_bundles, spec, top_k=5)

    # RAG: pass pool straight to listwise rerank — no retrieval needed.
    rag_input = [self._build_pool_entry(b) for b in pool_bundles]
    rag_picks = self.rag_rerank(query, rag_input)

    # Judge: same as before.
    profiles = {b["candidate"]["id"]: render_mini(b) for b in pool_bundles}
    suggested, reasoning = self.judge(query, rag_picks, det_picks, profiles)

    return ChatResponse(..., suggested=suggested, reasoning=reasoning)
```

Empty refinement (all prior candidates fail new constraint):
- `det_picks`, `rag_picks`, `suggested` are all empty.
- `reasoning` set to: `"No prior candidates matched the new constraint — try a broader follow-up or a fresh query."`
- Client gets a 200 response with empty lists; no exception.

### 4.4 `run_deterministic_on_pool` — helper in `app/deterministic_agent.py`

Tiny shim around existing `filter_and_score`. Same scoring math; just takes an explicit bundle list instead of the service-level full bundle cache.

Already exists effectively — `filter_and_score(bundles, spec, top_k)` is already pool-agnostic. No code change needed; just wire.

### 4.5 API-shape changes

**`ChatRequest`** — unchanged. Already has `conversation_id: str | None = None`.

**`ChatResponse`** — adds two fields:

```python
class ChatResponse(BaseModel):
    query: str
    conversation_id: str              # NEW — echoed so client can pass it next turn
    is_refinement: bool = False       # NEW — True iff this turn was a refinement
    parsed_spec: ParsedSpec
    rag_picks: list[CandidateResult]
    det_picks: list[CandidateResult]
    suggested: list[CandidateResult]
    reasoning: str
```

**`ParsedSpec`** — adds `is_refinement: bool = False`.

## 5. Storage, lifetime, and scale notes

- **Max memory:** each session holds one `ChatResponse` (~5-10 KB including candidate detail). 1000 concurrent sessions ≈ 10 MB. Fine.
- **TTL default:** 1800 s (30 min idle). Configurable via `SESSION_TTL_SECONDS` env var.
- **Cleanup cadence:** lazy on every `get`/`put`. No background thread.
- **Process restart:** all sessions are lost. Acceptable for a demo; documented in README.
- **Multi-worker:** we run uvicorn single-worker by default. If we ever scale horizontally, the `SessionStore` abstraction is swappable for Redis.

## 6. Testing plan

1. **`tests/test_session_store.py`** — unit tests for `get`/`put`/TTL expiry/thread safety.
2. **`tests/test_query_parser.py`** — extend: parser-with-prior-context sets `is_refinement=True` on a clearly refining query, `False` on a totally different one.
3. **`tests/test_search.py`** — extend `SearchService` tests: refinement path fetches the restricted pool, passes it to rerank/det, echoes `conversation_id`, stores new turn.
4. **`tests/test_routes_chat.py`** — extend: a second call with `conversation_id` from the first response routes through the refinement path.
5. **Live end-to-end** (manual): do the brief's exact example flow.

## 7. Response shape

```json
{
  "query": "Filter those to only people in Saudi Arabia",
  "conversation_id": "a3c8f2-...",
  "is_refinement": true,
  "parsed_spec": {
    "geography": { "values": ["SA"], "weight": 1.0, "required": false, "location_type": "current_or_nationality" },
    "temporality": "any",
    "is_refinement": true
  },
  "rag_picks": [ ... ≤ 5 from restricted pool ... ],
  "det_picks": [ ... ≤ 5 from restricted pool ... ],
  "suggested": [ ... ≤ 5 from restricted pool ... ],
  "reasoning": "Of the 5 prior candidates, 2 had AE nationality; ..."
}
```

## 8. Open items

None. Design is complete for Option A.

## 9. Cross-check against the brief

The brief's exact language:
> *"Support conversational context through a session or conversation ID, so follow-up queries can reference prior results (e.g., 'Filter those to only people based in Saudi Arabia')."*

- ✅ Session via `conversation_id`
- ✅ Follow-up queries can reference prior results
- ✅ Exact example ("Filter those to X") works via the narrow-refinement path

---
