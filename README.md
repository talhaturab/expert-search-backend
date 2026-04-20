# Expert Search — Applied AI Take-Home

FastAPI service that takes natural-language queries (*"Find me regulatory affairs experts with pharma experience in the Middle East"*) and returns a ranked shortlist of candidates from a Postgres database of ~10,000 expert profiles.

Two retrieval agents run in parallel and a judge cherry-picks the final top 5:

1. **RAG agent** — hybrid retrieval (cosine + BM25 with RRF fusion) over 3 natural-language probe texts per candidate, then a parallel per-candidate LLM rerank.
2. **Deterministic agent** — structured SQL scoring across 6 explicit dimensions (function, industry, geography, seniority, skills, languages) with LLM-assigned weights.
3. **Judge** — LLM sees both lists and cherry-picks the best 5 with reasoning.

Every result returned to the caller carries the four fields the brief asks for: **candidate_id · relevance score · match explanation · key highlights**.

**Status:** All endpoints live — `/ingest`, `/chat`, `/experts/{id}`, `/health`. Both agents (RAG + deterministic) plus the judge run end-to-end against a real Postgres DB of ~10K candidates and OpenRouter. See [`docs/superpowers/specs/2026-04-20-expert-search-design.md`](docs/superpowers/specs/2026-04-20-expert-search-design.md) for the full approved design.

---

## Quick start

```bash
# 1. Install deps
poetry install

# 2. Configure environment
cp .env.example .env
#   Fill in DATABASE_URL (your Postgres) and OPENROUTER_API_KEY
#   Default LLM_MODEL is anthropic/claude-haiku-4.5 for cheap dev

# 3. Run the API
poetry run uvicorn app.main:app --reload --port 8000

# 4. In another terminal — build the vector index
curl -s -X POST http://localhost:8000/ingest \
    -H "Content-Type: application/json" \
    -d '{"force": true}' | jq

# 5. Check readiness
curl -s http://localhost:8000/health | jq
```

Interactive API docs (Swagger) at `http://localhost:8000/docs`.

### CLI companion

Same core services, no HTTP round-trip:

```bash
poetry run python -m app.cli ingest --force
poetry run python -m app.cli chat "regulatory affairs experts in pharma in the Middle East"
```

Prints `ChatResponse` as formatted JSON to stdout.

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Liveness + readiness (DB reachable, Chroma populated, key set) |
| `POST` | `/ingest` | Build / rebuild the vector index from Postgres |
| `POST` | `/chat`   | Submit a NL query; return ranked experts |
| `GET`  | `/experts/{candidate_id}` | Full candidate profile as Markdown |

### `POST /ingest`

```json
{ "force": false }  // set true to re-embed even if index is already populated
```

Returns:

```json
{ "candidates_indexed": 200, "documents_written": 600, "duration_seconds": 161.3 }
```

By default indexing is capped at **200 candidates** (`INGEST_LIMIT` in `.env`) for fast dev iteration. Remove `INGEST_LIMIT` or set it to empty to index the full 10K.

### `POST /chat`

```json
{ "query": "regulatory affairs experts in pharma in the Middle East" }
```

Returns:

```json
{
  "query": "...",
  "parsed_spec": { /* function, industry, geography, seniority, weights, ... */ },
  "rag_picks":  [{ "candidate_id": "...", "rank": 1, "score": 85,
                   "match_explanation": "...", "highlights": ["..."] }, ...],
  "det_picks":  [{ "candidate_id": "...", "rank": 1, "score": 0.82,
                   "per_dim": { "industry": 1.0, "geography": 1.0, ... },
                   "match_explanation": "...", "highlights": ["..."] }, ...],
  "suggested":  [{ "candidate_id": "...", "rank": 1,
                   "match_explanation": "...", "highlights": ["..."] }, ...],
  "reasoning":  "..."
}
```

---

## Design summary

- **📖 Demo guide** (architecture + SVG diagrams + reasoning for every design choice): [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) — start here
- Approved design spec: [`docs/superpowers/specs/2026-04-20-expert-search-design.md`](docs/superpowers/specs/2026-04-20-expert-search-design.md)
- Conversation-context addition: [`docs/superpowers/specs/2026-04-21-conversation-context-design.md`](docs/superpowers/specs/2026-04-21-conversation-context-design.md)
- **Part 3 — Evaluation & Precision Thinking** (brief's written deliverable): [`docs/PART_3_EVALUATION.md`](docs/PART_3_EVALUATION.md)

### Vector database choice — **Chroma** (SQLite-backed, local)

- Zero infra: no Docker, no network service, persists to a local `.sqlite3` file
- Native metadata support — `{candidate_id, view}` is stored alongside each vector
- At 30K vectors our scale is well under what Chroma handles comfortably

At query time we **bypass Chroma's top-K API** entirely. We load all 30K embeddings into a numpy matrix once at startup and compute similarity with one matmul per query. This:

- Gives every candidate complete per-view scores (no "missing hits" problem from top-K cutoff)
- Runs in ~20 ms for 30K docs
- Keeps the code easy to reason about

### Embedding strategy — 3 probe texts per candidate

For each of the 10K candidates we render 3 natural-language probe texts (not raw concatenations):

- **summary** — elevator-pitch bio
- **work** — career narrative
- **skills_edu** — skills + education prose

Each is embedded separately → 30K vectors in one Chroma collection with metadata `view` ∈ {`summary`, `work`, `skills_edu`}. Natural-language prose aligns better with how queries are phrased than raw concatenation of structured fields.

### Query handling & ranking

1. **Query parser** (LLM, one call via structured outputs) → `ParsedSpec` — extracts `function`, `industry`, `geography`, `seniority`, `skills`, `languages`, `min_years_exp`, `temporality`, `view_weights` from NL. Expands entities (`"Middle East"` → ISO codes, `"pharma"` → `["Pharmaceuticals", "Biotechnology"]`), fixes typos, assigns weights, decides required-vs-soft.
2. **RAG agent**:
    - Full-scan cosine across 30K vectors → (10K × 3) score matrix
    - Full-scan BM25 across 30K texts → same shape
    - Aggregate per candidate using `view_weights` (query-weighted sum; `max` fallback)
    - **Reciprocal Rank Fusion** over vector & BM25 ranks (k=60, no normalization needed)
    - Take top-50 → parallel pointwise LLM rerank (one call per candidate via ThreadPoolExecutor, 16 workers) → sort by LLM score → top 5
3. **Deterministic agent**: 6 dimension scoring functions, hard-filter pool, weighted sum → top 5
4. **Judge**: one LLM call seeing both lists, cherry-picks the final 5

### Model choice — LLM

Default `LLM_MODEL=anthropic/claude-haiku-4.5` (via OpenRouter) for cheap dev iteration. Trade up to `anthropic/claude-sonnet-4.6` for production quality — same API, no code changes, ~3× more expensive.

Structured outputs are done via the OpenAI **Responses API** (`client.responses.parse(text_format=PydanticModel)`) — the non-beta path. Pydantic models → JSON Schema → provider returns validated JSON → SDK parses into typed instance. No manual JSON parsing, no fence stripping.

**Schema-compatibility caveats** (discovered live against Anthropic through OpenRouter):
- Don't use `dict[Literal[...], float]` — Pydantic emits `propertyNames`, which Anthropic rejects. Use an explicit `BaseModel` for fixed-key maps.
- Don't use `Field(ge=..., le=...)` on int/float fields — Pydantic emits `minimum`/`maximum`, which Anthropic rejects. Clamp in code after parsing.

### Embedding model

`openai/text-embedding-3-small` (1536-dim) via OpenRouter — $0.02 / 1M tokens, high quality for the price. Ingestion cost is negligible: 30K short probe texts × ~150 tokens ≈ 4.5M tokens ≈ $0.09 for the full 10K index.

---

## Project layout

```
app/
├── main.py                # FastAPI app factory
├── cli.py                 # CLI companion (chat, ingest subcommands)
├── config.py              # pydantic-settings Settings (loads from .env)
├── models.py              # All Pydantic request/response/shared models
├── db.py                  # Postgres loaders
├── vocabulary.py          # Load distinct DB values for parser grounding
├── profile_builder.py     # Render candidate profiles (mini + full markdown)
├── probe_texts.py         # Render summary/work/skills_edu views for embedding
├── embeddings.py          # OpenRouter embedding client
├── llm.py                 # OpenRouter LLM client (chat + chat_structured via Responses API)
├── chroma_store.py        # Chroma wrapper
├── bm25_index.py          # rank_bm25 wrapper
├── ingest.py              # Offline indexer orchestration
├── query_parser.py        # NL -> ParsedSpec via one structured LLM call
├── scoring.py             # 6 per-dim deterministic scoring functions
├── rag_agent.py           # retrieve_candidates + parallel pointwise rerank
├── deterministic_agent.py # Hard-filter + weighted-sum scoring orchestration
├── judge.py               # Single-LLM cherry-pick over both agents' picks
├── search.py              # SearchService orchestrator (DI per stage)
└── routes/
    ├── health.py
    ├── ingest.py
    ├── chat.py            # Lazy-loaded SearchService cached via @lru_cache
    └── experts.py

tests/                    # Pytest suite (mocked for unit, real services marked @pytest.mark.integration)
docs/superpowers/specs/   # Design spec
docs/superpowers/plans/   # Implementation plan
```

---

## Conversational context

Follow-up queries can reference prior results via `conversation_id`:

```bash
# First turn — server returns a conversation_id in the response
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "regulatory affairs experts in pharma in the Middle East"}' \
  | jq '{conversation_id, suggested}'

# Second turn — reuse the conversation_id to narrow the prior result set
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "filter those to only people in Saudi Arabia", "conversation_id": "<paste from above>"}' \
  | jq '{is_refinement, suggested}'
```

The second call takes the prior turn's `suggested` candidates (up to 5) and re-scores them against the new constraint instead of running a fresh full-scan search. Sessions live 30 minutes in-memory and are lost on process restart.

Only **narrowing** follow-ups are supported in this iteration (*"filter those to..."*, *"among them, only..."*, *"narrow to..."*). Broadening follow-ups (*"show me more like these"*) and explanation questions (*"why did you pick #3?"*) are not.

---

## Testing

```bash
# Unit tests — fast, no external services
poetry run pytest -v -m "not integration"

# Integration tests — hit real Postgres / OpenRouter
poetry run pytest -v -m integration
```

---

## Out of scope for this iteration

- **HyDE** (hypothetical document embeddings at query time) — the `HYDE_ENABLED` flag exists in `.env` but the step is not wired into `SearchService`. Easy to add later if retrieval quality on vague queries proves insufficient.
- **Broadening / explanation follow-ups** in conversation context — narrow-refinement only.
