# Expert Search Service — Design

**Date:** 2026-04-20
**Status:** Approved design; awaiting implementation plan
**Scope:** Natural-language candidate search over the `candidate_profiles` Postgres DB (10,120 candidates)

---

## 1. Goals

Build a service that accepts a natural-language query (e.g., *"Find me regulatory affairs experts in pharma in the Middle East"*) and returns the top 5 best-matching candidates.

Core requirements:

- **Weighted signal scoring** — queries emphasizing seniority, geography, and function must be weighted differently than queries asking for "junior data engineers anywhere." Ranking must reflect query intent, not embedding distance alone.
- **Comparative architecture** — two independent retrieval agents (one RAG-based, one deterministic-SQL-based) plus a judge that cherry-picks the final 5 across both. This lets us A/B-compare the approaches on the same queries AND produce a best-of-both suggestion per query.
- **Explainability** — per-candidate scores (with per-dimension breakdowns for the deterministic agent) and a reasoning string from the judge.
- **Graceful handling** — hard filters that return explicit emptiness (with reason) rather than silently degrading to irrelevant results.

## 2. Non-goals

- Learning-to-rank / trained ranking models (no labeled data available)
- Fine-tuning embedding models
- Scaling beyond ~100K candidates (30K embeddings fit in RAM; no ANN index needed)
- Personalization / preference learning
- A full UI (CLI + one HTTP endpoint is sufficient to demo both)

## 3. High-level architecture

```
Query
  │
  ▼
┌──────────────────────┐
│  Query Parser (LLM)  │──▶ structured spec {dims, weights, expansions, required flags}
└──────────┬───────────┘
           │
   ┌───────┴───────┐
   │               │
   ▼               ▼
┌────────┐   ┌──────────────┐
│ RAG    │   │ Deterministic│
│ Agent  │   │ Agent (SQL)  │
└───┬────┘   └──────┬───────┘
    │               │
  rag_picks[5]   det_picks[5]
    │               │
    └───────┬───────┘
            ▼
      ┌──────────┐
      │  Judge   │  ──▶  suggested[5] + reasoning
      └──────────┘
            │
            ▼
  Response: { parsed_spec, rag_picks, det_picks, suggested, reasoning }
```

**Three LLM calls per query** (4 with optional HyDE):
1. Query Parser (shared upstream)
2. RAG listwise rerank
3. Judge
4. *(Optional)* HyDE inside RAG

## 4. Data & indexing

### Source DB

Postgres `candidate_profiles` (read-only credentials). Schema fully mapped in `SCHEMA.md`:
- 10,120 candidates
- 24,635 work experiences (FK to companies → industry, country)
- 20,171 education records
- 50,638 candidate_skills (FK to skills → skill_categories)
- 25,276 candidate_languages (FK to languages + proficiency_levels)

### Offline indexing (run once)

For each of the 10,120 candidates, synthesize **3 natural-language probe texts** (not raw concatenations — actual prose generated from a template):

**summary** — elevator-pitch bio (~80 tokens)
> *"Sara Ali is an Engineering Manager with 16 years of experience. Currently Engineering Manager Research Scientist at IDM Brokerage House (Finance) in Czech Republic. Based in Philadelphia, US; Latvian nationality."*

**work** — work-history narrative (~250 tokens)
> *"Sara's career includes roles as Engineering Manager Research Scientist at IDM Brokerage House (2025–present), where she spearheaded migration to analysis, reducing costs by 58%. Previously: ..."*

**skills_edu** — skills + education narrative (~150 tokens)
> *"Sara is an expert in Brand Marketing (9y) and Analysis (6y). Holds an Associate's in Petroleum and Natural Gas Engineering from Union High School (2007–2010, B-)."*

**Why probe texts, not raw concat:** embeddings for short, natural-language prose align much better with query embeddings than they do with raw UUID-heavy or keyword-salad text. This is the "view design" part of V2++.

### Storage

- **Chroma** (local, SQLite-backed) — stores 30,360 embeddings with `{candidate_id, view, source_text}` metadata
- **numpy matrix cache** — at service startup, load all embeddings from Chroma into a `(30360, D)` float32 array plus a parallel `candidate_ids[30360]` array. All query-time similarity is computed in-memory via matmul.
- **Source texts** also cached in memory for BM25 (`rank_bm25` indexes them on startup)

Embedding model: `text-embedding-3-small` via OpenAI/OpenRouter (default; swappable).

## 5. Runtime pipeline — per query

### 5.1 Query Parser (shared)

Single LLM call. Takes NL query, returns structured spec:

```json
{
  "function":   { "values": ["Regulatory Affairs", "Compliance"], "weight": 0.35, "required": true },
  "industry":   { "values": ["Pharmaceuticals", "Biotechnology"], "weight": 0.30, "required": true },
  "geography":  { "values": ["AE", "SA", "QA", "BH", "KW", "OM", "EG", "JO", "LB"],
                  "weight": 0.20, "required": false,
                  "location_type": "current_or_nationality" },
  "seniority":  { "levels": ["senior", "executive"], "weight": 0.10, "required": false },
  "skills":     { "values": [], "weight": 0.0, "required": false },
  "languages":  { "values": [], "required_proficiency": null, "weight": 0.0, "required": false },
  "min_years_exp": null,
  "temporality": "any",
  "view_weights": { "summary": 0.3, "work": 0.5, "skills_edu": 0.2 }
}
```

Parser responsibilities:

1. **Entity extraction** from NL (function, industry, geography, seniority, skills, languages)
2. **Synonym/entity expansion** — `"Middle East"` → country codes; `"pharma"` → related industry list; `"CPO"` → disambiguated title candidates given context
3. **Typo/misspelling normalization** — `"pharmacuetical"` → `"Pharmaceutical"`
4. **Weight assignment** — based on emphasis in query phrasing
5. **Required-vs-nice** — hard filter (`required: true`) only when query uses "must," "only," "specifically" or equivalent; soft (`required: false`) by default
6. **View weights** — for the RAG agent, what mix of (summary, work, skills_edu) should the aggregation emphasize given this query type
7. **`temporality`** — `"current"` | `"past"` | `"any"` — triggered by phrasing like `"former CPO"` or `"currently a VP"`

#### Weighting heuristics (user-provided)

**This is the one place where user domain input shapes the solution.** The parser prompt includes a block of 5-10 if-then rules the user writes. Example rules (to be replaced by user's):

```
- If query mentions "former [role]" or "ex-[role]": temporality=past, seniority.weight >= 0.3
- If query mentions "junior" or "entry-level" or "new grad": seniority=[junior], geography.required=false
- If query specifies a company: companies becomes a hard filter
- If query mentions "must have N+ years": min_years_exp=N, required=true
```

These rules live in the parser's system prompt and are the main lever the user has for tuning how queries map to weights.

### 5.2 RAG Agent

**Goal:** semantic retrieval — top 5 candidates by combined vector + BM25 fit.

**Flow:**

1. **(Optional) HyDE** — LLM writes a hypothetical ideal candidate profile from the query. Used as the retrieval query instead of the raw NL query, because it lives in the same "language space" as real profiles and aligns better via cosine.
2. **Embed** the query (or HyDE output) → `query_vec` shape `(D,)`
3. **Full-scan vector similarity**: `sims = all_embeddings @ query_vec` → shape `(30360,)`. One matmul, ~20ms.
4. **Full-scan BM25**: `bm25_scores = bm25_index.get_scores(query_tokens)` → shape `(30360,)`. ~10ms.
5. **Reshape both** to `(10120, 3)` — rows=candidates, cols=views
6. **Aggregate per candidate** on each axis separately. **Query-weighted sum** using `view_weights` from the parser when available; fall back to `max` otherwise:

   ```python
   if spec.view_weights and any(w > 0 for w in spec.view_weights.values()):
       # Query-weighted sum — parser supplied view weights
       agg_vec[c]  = w_sum*sims[c, 0]  + w_work*sims[c, 1]  + w_edu*sims[c, 2]
       agg_bm25[c] = w_sum*bm25[c, 0]  + w_work*bm25[c, 1]  + w_edu*bm25[c, 2]
   else:
       # Fallback — strongest-view match (no parser weights available)
       agg_vec[c]  = max(sims[c, 0],  sims[c, 1],  sims[c, 2])
       agg_bm25[c] = max(bm25[c, 0], bm25[c, 1], bm25[c, 2])
   ```

   **Rationale:** when the parser confidently weights views based on query shape (e.g., work-heavy for `"former CPO at ..."`, skills-heavy for `"Python expert"`), retrieval uses that domain signal. When `view_weights` is missing or all-zero, `max` is a safe fallback — rewards any strong-view match without requiring tuning constants.
7. **Fuse** via Reciprocal Rank Fusion (RRF, k=60):
   ```
   rank_vec  = argsort_desc(agg_vec)
   rank_bm25 = argsort_desc(agg_bm25)
   rrf[c]    = 1/(60 + rank_vec[c]) + 1/(60 + rank_bm25[c])
   ```
8. **Top 50 candidates** by `rrf`
9. **Listwise LLM rerank** — single LLM call receiving all 50 candidate mini-profiles + query. Returns top 5, each with:
   - `rank` (1–5) and `score` (the RRF input score)
   - `match_explanation` — 1–2 sentences on why this candidate ranked here relative to the others
   - `highlights` — 2–4 bullet-style proof-points drawn from the candidate's profile that are relevant to the query (e.g., *"12y Regulatory Affairs at Pfizer"*, *"based in Dubai, UAE"*)

   The LLM makes relative comparisons, not independent 0-100 scoring.

**Why these choices:**

- **Probe-text views over raw concatenations**: aligns query and candidate embeddings in the same linguistic register.
- **Full-scan over top-K retrieval**: gives every candidate complete per-view scores; no candidate silently dropped because their hits fell outside a top-K window.
- **RRF over weighted sum of raw scores**: normalization-free; robust across vector/BM25 distribution differences; well-established default.
- **Listwise rerank over pointwise**: 1 LLM call instead of 50; relative comparisons empirically outperform independent scoring for rerank tasks.

### 5.3 Deterministic Agent

**Goal:** structured scoring — top 5 candidates by weighted sum over six explicit dimensions.

**Flow:**

1. **Hard-filter pool**: start from all candidates, filter to those matching every `required: true` dimension. If pool is empty → return `det_picks: []` with a reason string.
2. **Score each candidate on each mentioned dimension** (see Section 6 for formulas).
3. **Weighted sum**: `final = Σ (dim.weight * dim.score)`
4. **Top 5** by `final` → `det_picks`, each with:
   - `score` (weighted sum) and per-dim breakdown
   - `match_explanation` — auto-generated from the top-contributing dimensions (e.g., *"Strong match on industry (1.0) and function (0.91); weaker on geography (0.0)."*)
   - `highlights` — deterministically computed — pick 2–4 rows from `work_experience`/`candidate_skills` that map to the spec's matching dimensions (e.g., matching-industry current job, highest-years matching skill, matching-country location)

**No LLM calls in this agent.** All scoring, explanations, and highlights are computed from the spec + candidate data.

### 5.4 Judge

**Goal:** cherry-pick the best 5 across both agents' outputs.

**Input to one LLM call:**
- Original query
- `rag_picks` (5 candidates with their RRF-ranked scores)
- `det_picks` (5 candidates with final scores + per-dim breakdown)
- Full profile markdown for each unique candidate (≤10 profiles — reuses the existing `profile.py` renderer)

**Output:**

```json
{
  "suggested": [
    { "candidate_id": "cid-1", "rank": 1,
      "match_explanation": "...", "highlights": ["...", "..."] },
    ...
  ],
  "reasoning": "Cid-1 appears in both lists — strong pharma + ME signal. Cid-2 from RAG only; semantic match on 'regulatory affairs' in a job description the structured agent couldn't see. Cid-3 has the deepest tenure (11y pharma vs. 4y average). [...]"
}
```

Each suggested candidate's `match_explanation` and `highlights` are inherited from whichever agent selected them (RAG's rerank LLM output or Deterministic's computed output). The judge may overwrite these if it has a better framing, but the default is passthrough.

**Prompt shape:**
```
You are comparing candidate recommendations from two search agents for this query:
"{query}"

Agent 1 (RAG — semantic):                [profiles + RRF scores]
Agent 2 (Deterministic — structured):    [profiles + scores + per-dim breakdown]

Pick the 5 candidates overall that best match the query. You may cherry-pick
from either list, or favor one if its picks are clearly stronger. Explain
your reasoning briefly.
```

## 6. Deterministic scoring dimensions

Every score is normalized to `[0, 1]`.

### Function

- **Signal source:** `work_experience.job_title`, `candidates.headline`
- **Score:** best fuzzy match (`pg_trgm` trigram similarity) between `target_function` and any of the candidate's job titles or headline
- **Recency weighting:** current job contributes 1.0×; past jobs discounted by years-since-end
- **Formula sketch:**
  ```
  fn_score = max over (title, recency) of (trgm_sim(title, target) * recency_factor)
  ```

### Industry

- **Signal source:** `work_experience → companies.industry`
- **Score:** fraction of candidate's total career years spent in matching industries
- **Formula sketch:**
  ```
  industry_score = sum(years_per_job where industry ∈ target) / sum(years_per_job)
  ```
- **Partial credit:** target list may be weighted (`"Pharmaceuticals": 1.0, "Biotechnology": 0.7`) — LLM parser produces these weights

### Geography

- **Signal source:** `candidates.city → cities.country`, `candidates.nationality`, `companies.country` (via work_experience)
- **Score:** depends on `location_type` from spec:
  - `"current"`: candidate's current city's country ∈ target → 1.0, else 0
  - `"current_or_nationality"`: 1.0 if either matches, 0 otherwise
  - `"historical"`: max over jobs with company country ∈ target (recency-weighted)

### Seniority

- **Signal source:** `candidates.years_of_experience` + title keyword scan
- **Years → level mapping:**
  - `0–3` years → junior
  - `3–8` years → mid
  - `8–15` years → senior
  - `15+` years → executive
- **Title boost:** current job title contains `"Chief" | "VP" | "Head" | "Director" | "Principal"` → upgrade one level
- **Score:** 1.0 if candidate's level ∈ target levels; 0.5 if adjacent; 0 otherwise

### Skills

- **Signal source:** `candidate_skills → skills.name`
- **Score:** hits-over-target with years bonus
- **Formula sketch:**
  ```
  matched = {s for s in target if s ∈ candidate.skills}
  years_factor = mean(years_of_experience[s] for s in matched) / 10  # cap at 1
  skills_score = (|matched| / |target|) * (0.7 + 0.3 * years_factor)
  ```

### Languages

- **Signal source:** `candidate_languages → languages, proficiency_levels`
- **Score:** fraction of target languages held at or above required proficiency

## 7. Response shape

Every returned expert carries four brief-required fields: **profile (id) · relevance score · match explanation · key highlights**.

```json
{
  "query": "Find me regulatory affairs experts in pharma in the Middle East",
  "parsed_spec": { /* full spec as produced by parser */ },

  "rag_picks": [
    { "candidate_id": "...", "rank": 1, "score": 0.0347,
      "match_explanation": "Work description explicitly mentions regulatory submissions in GCC markets...",
      "highlights": ["10y pharma at Pfizer", "based in Dubai", "FDA + GCC submission lead"] },
    ...
  ],

  "det_picks": [
    { "candidate_id": "...", "rank": 1, "score": 0.82,
      "per_dim": { "function": 0.91, "industry": 1.0, "geography": 0.0, "seniority": 0.7 },
      "match_explanation": "Strong on industry (1.0) and function (0.91); weak on geography (0.0).",
      "highlights": ["Current: Regulatory Affairs Director, Novartis", "12y pharma", "Based in Switzerland"] },
    ...
  ],

  "suggested": [
    { "candidate_id": "...", "rank": 1,
      "match_explanation": "...", "highlights": ["...", "..."] },
    ...
  ],

  "reasoning": "..."
}
```

Full candidate profile Markdown is available via the existing `profile.py` renderer; we can also expose it via `GET /experts/{candidate_id}` (see §9).

## 8. Stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.10+ | Existing `pyproject.toml` / Poetry env |
| DB access | `psycopg2-binary` (already installed) | Matches existing scripts |
| Vector store | **Chroma** (local, SQLite-backed) | Clean API, persistent, metadata, no Docker |
| BM25 | `rank_bm25` (Python) | Simple, controllable tokenization |
| Numpy | for full-scan matmul | 30K × 1536 matrix, ~20ms per query |
| LLM provider | **OpenRouter** (user has creds) | One provider, many models |
| LLM (parser, judge, rerank) | Claude 3.5 Sonnet (default) | Strong reasoning; swappable |
| Embeddings | `text-embedding-3-small` (via OpenAI/OpenRouter) | Cheap, high quality |
| API | **FastAPI** (required by brief) with `/chat`, `/ingest`, `/health` | Swagger-ready, Pydantic-native |
| Request/response models | **Pydantic v2** (required by brief) | Typed schemas for every endpoint |
| Configuration | Environment variables via `.env` + `pydantic-settings` | No hardcoded secrets; `.env.example` committed |
| CLI | `argparse` | Matches existing repo style |

## 9. Form factor — FastAPI endpoints

Brief requires a FastAPI backend. All endpoints use Pydantic v2 models for request/response; Swagger is auto-generated at `/docs`.

### Endpoints (minimum)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | Trigger the offline indexing pipeline (idempotent — re-running rebuilds the Chroma collection) |
| `POST` | `/chat` | Accept a NL query, return the full response (rag_picks + det_picks + suggested + reasoning) |
| `GET`  | `/health` | Liveness + readiness check (reports whether Chroma is populated, Postgres is reachable, LLM key is set) |
| `GET`  | `/experts/{candidate_id}` | Return the full rendered profile for a candidate (reuses `profile.py`) — useful for the UI/caller to drill down after a `/chat` response |

### Pydantic models (sketch)

```python
# Request models
class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None  # accepted but no-op for now; see §12 deferred

class IngestRequest(BaseModel):
    force: bool = False  # re-embed even if Chroma already populated

# Core shared models
class ParsedSpec(BaseModel):
    function: DimensionSpec | None
    industry: DimensionSpec | None
    geography: GeoSpec | None
    seniority: SenioritySpec | None
    skills: SkillsSpec | None
    languages: LanguagesSpec | None
    min_years_exp: int | None
    temporality: Literal["current", "past", "any"]
    view_weights: dict[Literal["summary", "work", "skills_edu"], float] | None

class CandidateResult(BaseModel):
    candidate_id: str
    rank: int
    score: float
    match_explanation: str
    highlights: list[str]
    per_dim: dict[str, float] | None = None  # only populated for det_picks

# Response models
class ChatResponse(BaseModel):
    query: str
    parsed_spec: ParsedSpec
    rag_picks: list[CandidateResult]
    det_picks: list[CandidateResult]
    suggested: list[CandidateResult]
    reasoning: str

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, bool]  # {"chroma_populated": True, "postgres_reachable": True, ...}
```

### Configuration (env-driven, no hardcoded secrets)

`.env.example` committed:

```
DATABASE_URL=postgresql://developer:YOUR_PASSWORD@34.79.32.228:5432/candidate_profiles
OPENROUTER_API_KEY=sk-or-...
CHROMA_PERSIST_PATH=./data/chroma
LLM_MODEL=anthropic/claude-3.5-sonnet
EMBEDDING_MODEL=openai/text-embedding-3-small
```

Loaded via `pydantic-settings` (`Settings(BaseSettings)`).

### CLI companion (optional convenience)

```bash
poetry run python -m app.cli chat "find me regulatory affairs experts in pharma in the Middle East"
poetry run python -m app.cli ingest --force
```

Both the CLI and the FastAPI routes call the same core functions (`chat_service.search()`, `ingest_service.run()`) — no logic duplication.

## 10. Testing / evaluation approach

No labeled ground truth → no offline metric. Evaluation is **query-set + eyeball + comparison**:

1. **Fixed benchmark queries** (5–10) covering different shapes:
   - Hard-criteria specific ("former CPO at Saudi petrochemical")
   - Fuzzy / general ("regulatory affairs experts in pharma in the Middle East")
   - Underspecified ("junior data engineers anywhere")
   - Edge cases (typos, ambiguous terms, "MENA" vs "Middle East")
2. **Per query, inspect all three lists** — `rag_picks`, `det_picks`, `suggested` — plus the `reasoning`
3. **Agreement metric** (informal): `|rag_picks ∩ det_picks|`; `|rag_picks ∩ suggested|`; `|det_picks ∩ suggested|`. High agreement + sensible reasoning = system is working.
4. **Per-dim score inspection** on `det_picks` for debuggability — a wrong answer should surface a dim with an unexpected value.

## 11. Stack diagram — at a glance

```
┌──────────────┐            ┌──────────────────────────────┐
│   Postgres   │──read──────│ Candidate data               │
│ (read-only)  │            │ (tables, queried at runtime) │
└──────────────┘            └──────────────┬───────────────┘
                                           │
                          ┌────────────────┴──────────────────┐
                          │                                   │
             ┌────────────▼───────────┐         ┌─────────────▼────────────┐
             │ Offline indexer        │         │ Deterministic Agent      │
             │ • render probe texts   │         │ (SQL + Python scoring)   │
             │ • embed 30K docs       │         └─────────────▲────────────┘
             │ • populate Chroma      │                       │
             └────────────┬───────────┘                       │
                          │                                   │
                          ▼                                   │
             ┌────────────────────────┐                       │
             │ Chroma (+ numpy cache) │◀────── RAG Agent ─────┘
             │ + rank_bm25 index      │        (semantic + BM25 + rerank)
             └────────────────────────┘
                          │
                          ▼
                ┌────────────────┐
                │ Judge (LLM)    │
                └────────────────┘
                          │
                          ▼
                    Response JSON
```

## 12. Open items

### Active (to resolve before or during implementation)

- **Weighting heuristics for the query parser** — user to supply 5–10 if-then rules (Section 5.1). Spec intentionally leaves this for the user's domain input.
- **Embedding model choice** — default `text-embedding-3-small`; can swap to any OpenRouter-compatible embedding.
- **LLM model choice** — default Claude 3.5 Sonnet for parser/judge/rerank; cost-sensitive deployments may use Haiku for parser only.
- **HyDE on/off** — default on; easy to toggle via config.

### Deferred (explicitly out of scope for this iteration)

- **Conversational context / follow-up refinement** (brief Part 2 item). `ChatRequest` accepts `conversation_id` to preserve the schema, but the value is currently a no-op. Follow-up queries like *"Filter those to Saudi Arabia only"* are not handled. Deferred to a later iteration.
- **Part 3 — Evaluation & Precision Thinking** (ground-truth dataset design, precision metrics, failure analysis). Written deliverable only, no code. Explicitly out of scope for this iteration.

## 13. Cross-check against take-home brief

**Performed 2026-04-20** against `takehome_assessment.docx`. Findings summary:

### Requirements satisfied by this spec

| Brief requirement | Where addressed |
|---|---|
| FastAPI backend (hard requirement) | §8, §9 |
| `POST /ingest` endpoint | §9 |
| `POST /chat` endpoint (NL query, ranked experts) | §9 |
| `GET /health` endpoint | §9 |
| Query transformation / decomposition (structured signals) | §5.1 parser |
| Scoring + re-ranking beyond raw vector similarity | §5.2 (RAG rerank) + §5.3 (deterministic) |
| Deliberate signal weighting | §5.3, §6 + query parser `weight` fields |
| Per-result match explanation | §7, §5.2, §5.3 |
| Per-result key highlights | §7, §5.2, §5.3 |
| Structured JSON responses, Pydantic models | §9 |
| Vector DB choice with justification | §8 (Chroma) |
| OpenRouter LLM key, model-choice rationale | §8, README (to be produced) |
| Environment variables for all secrets | §9 (`.env.example`) |
| `pyproject.toml` + minimal setup | Existing Poetry env |

### Deferred (not implemented now, see §12)

- **Conversational context / `conversation_id`** — schema accepts it, behavior is no-op.
- **Part 3 (Evaluation & Precision design doc)** — explicitly out of scope for this iteration.

### Design exceeds the brief (noted for the reader)

- Our **multi-agent architecture** (RAG + Deterministic + Judge) is richer than the brief's "scoring and re-ranking layer." We keep it because it gives a natural A/B comparison and a judge-synthesized best-of-both output. If simplification is ever desired, the deterministic agent and the judge can be dropped and only RAG-with-rerank ships.
- Response includes `rag_picks` and `det_picks` side-by-side (not required by brief). These are additive and don't break the required `suggested` list.

**Conclusion:** spec is consistent with the brief and ready for implementation planning.

---
