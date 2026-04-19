# Expert Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI backend that accepts natural-language queries and returns the top 5 candidates ranked by hybrid retrieval (RAG + deterministic SQL scoring) with a judge-synthesized best-of-both answer.

**Architecture:** Three pipelines per query — (1) LLM query parser produces a structured spec; (2) RAG agent retrieves via full-scan vector + BM25 with RRF fusion, then listwise LLM rerank; (3) Deterministic agent filters + scores SQL-side across six weighted dimensions. Judge LLM cherry-picks the final 5 across both lists.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, Chroma (SQLite-backed), rank_bm25, psycopg2, OpenRouter (OpenAI-compatible), numpy.

**Reference spec:** [`docs/superpowers/specs/2026-04-20-expert-search-design.md`](../specs/2026-04-20-expert-search-design.md)

---

## File structure

```
app/
├── __init__.py
├── main.py                # FastAPI application & router wiring
├── cli.py                 # CLI companion (chat, ingest subcommands)
├── config.py              # pydantic-settings (.env loading)
├── models.py              # All Pydantic request/response/shared models
├── db.py                  # Postgres connection + candidate loaders
├── profile_builder.py     # Adapts existing profile.py for in-service use
├── probe_texts.py         # Generate 3 natural-language views per candidate
├── embeddings.py          # OpenRouter embedding wrapper
├── llm.py                 # OpenRouter LLM wrapper (chat completions)
├── chroma_store.py        # Chroma collection management
├── bm25_index.py          # rank_bm25 wrapper
├── ingest.py              # Offline indexing orchestration
├── query_parser.py        # NL query → ParsedSpec (LLM call)
├── hyde.py                # Hypothetical Document Embeddings (LLM call)
├── scoring.py             # 6 deterministic scoring dimension functions
├── rag_agent.py           # Full-scan retrieval + RRF + listwise rerank
├── deterministic_agent.py # Hard-filter pool + weighted-dim scoring
├── judge.py               # Judge LLM call
├── search.py              # End-to-end orchestrator (used by /chat and CLI)
└── routes/
    ├── __init__.py
    ├── chat.py            # POST /chat
    ├── ingest.py          # POST /ingest
    ├── health.py          # GET  /health
    └── experts.py         # GET  /experts/{candidate_id}

tests/
├── conftest.py            # Shared fixtures
├── test_models.py
├── test_probe_texts.py
├── test_scoring.py
├── test_query_parser.py
├── test_rag_agent.py
├── test_deterministic_agent.py
├── test_judge.py
├── test_search.py
└── integration/
    └── test_chat_endpoint.py

.env.example
README.md
```

---

## Phase 0 — Foundation

### Task 1: Dependencies + config scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `.env.example`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add dependencies via Poetry**

```bash
poetry add chromadb rank_bm25 openai fastapi uvicorn pydantic pydantic-settings tenacity
poetry add --group dev pytest pytest-asyncio httpx
```

Expected: lockfile updates cleanly.

- [ ] **Step 2: Create the app package**

Create empty `app/__init__.py`:

```python
```

Create `tests/__init__.py`:

```python
```

- [ ] **Step 3: Write `.env.example`**

Create `.env.example`:

```
# Postgres (read-only creds provided in db_connection.txt)
DATABASE_URL=postgresql://developer:REPLACE_ME@34.79.32.228:5432/candidate_profiles

# OpenRouter API key (see openrouter.txt)
OPENROUTER_API_KEY=sk-or-v1-REPLACE_ME

# Storage paths (relative to repo root is fine)
CHROMA_PERSIST_PATH=./data/chroma

# Models
LLM_MODEL=anthropic/claude-3.5-sonnet
EMBEDDING_MODEL=openai/text-embedding-3-small

# Retrieval
HYDE_ENABLED=true
RAG_TOP_K=50
DETERMINISTIC_TOP_K=5
FINAL_TOP_K=5
```

- [ ] **Step 4: Write config tests first**

Create `tests/test_config.py`:

```python
import os
from app.config import Settings


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@host:5432/db")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", "/tmp/chroma")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

    s = Settings()

    assert s.database_url.endswith("/db")
    assert s.openrouter_api_key == "sk-or-test"
    assert s.chroma_persist_path == "/tmp/chroma"
    assert s.hyde_enabled is True  # default
    assert s.rag_top_k == 50       # default


def test_settings_defaults_for_optional_fields(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")

    s = Settings()

    assert s.hyde_enabled is True
    assert s.rag_top_k == 50
    assert s.deterministic_top_k == 5
    assert s.final_top_k == 5
```

- [ ] **Step 5: Run tests — expect failure**

```bash
poetry run pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.config'`.

- [ ] **Step 6: Implement `app/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    openrouter_api_key: str
    chroma_persist_path: str = "./data/chroma"
    llm_model: str = "anthropic/claude-3.5-sonnet"
    embedding_model: str = "openai/text-embedding-3-small"

    hyde_enabled: bool = True
    rag_top_k: int = 50
    deterministic_top_k: int = 5
    final_top_k: int = 5


def get_settings() -> Settings:
    return Settings()  # re-read env each call; fine for this scale
```

- [ ] **Step 7: Run tests — expect pass**

```bash
poetry run pytest tests/test_config.py -v
```

Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git init  # if not already
git add app/__init__.py app/config.py tests/__init__.py tests/test_config.py .env.example pyproject.toml poetry.lock
git commit -m "feat(config): pydantic-settings with .env loading"
```

---

### Task 2: Pydantic models

**Files:**
- Create: `app/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write model tests first**

Create `tests/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from app.models import (
    DimensionSpec, GeoSpec, SenioritySpec, SkillsSpec, LanguagesSpec,
    ParsedSpec, CandidateResult, ChatRequest, ChatResponse,
    IngestRequest, HealthResponse,
)


def test_parsed_spec_minimal():
    spec = ParsedSpec(temporality="any")
    assert spec.function is None
    assert spec.industry is None
    assert spec.temporality == "any"
    assert spec.view_weights is None


def test_parsed_spec_full():
    spec = ParsedSpec(
        function=DimensionSpec(values=["Regulatory Affairs"], weight=0.35, required=True),
        industry=DimensionSpec(values=["Pharmaceuticals"], weight=0.30, required=True),
        geography=GeoSpec(values=["AE", "SA"], weight=0.20, required=False,
                          location_type="current_or_nationality"),
        seniority=SenioritySpec(levels=["senior"], weight=0.10, required=False),
        temporality="any",
        view_weights={"summary": 0.3, "work": 0.5, "skills_edu": 0.2},
    )
    assert spec.function.values == ["Regulatory Affairs"]
    assert spec.geography.location_type == "current_or_nationality"


def test_parsed_spec_invalid_temporality():
    with pytest.raises(ValidationError):
        ParsedSpec(temporality="future")


def test_candidate_result():
    r = CandidateResult(
        candidate_id="c-1", rank=1, score=0.82,
        match_explanation="Strong industry match.",
        highlights=["12y pharma", "based in Dubai"],
    )
    assert r.rank == 1
    assert r.per_dim is None


def test_chat_request_default_conversation_id_is_none():
    req = ChatRequest(query="find pharma experts")
    assert req.conversation_id is None


def test_chat_response_shape():
    resp = ChatResponse(
        query="q",
        parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[],
        det_picks=[],
        suggested=[],
        reasoning="empty",
    )
    assert resp.query == "q"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
poetry run pytest tests/test_models.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `app/models.py`**

```python
from typing import Literal
from pydantic import BaseModel, Field


# ---------- Parsed Spec (query parser output) ----------

class DimensionSpec(BaseModel):
    values: list[str] = Field(default_factory=list)
    weight: float = 0.0
    required: bool = False


class GeoSpec(DimensionSpec):
    location_type: Literal["current", "current_or_nationality", "historical"] = "current_or_nationality"


class SenioritySpec(BaseModel):
    levels: list[Literal["junior", "mid", "senior", "executive"]] = Field(default_factory=list)
    weight: float = 0.0
    required: bool = False


class SkillsSpec(BaseModel):
    values: list[str] = Field(default_factory=list)
    weight: float = 0.0
    required: bool = False


class LanguagesSpec(BaseModel):
    values: list[str] = Field(default_factory=list)
    required_proficiency: Literal["Beginner", "Intermediate", "Fluent", "Native"] | None = None
    weight: float = 0.0
    required: bool = False


ViewName = Literal["summary", "work", "skills_edu"]


class ParsedSpec(BaseModel):
    function: DimensionSpec | None = None
    industry: DimensionSpec | None = None
    geography: GeoSpec | None = None
    seniority: SenioritySpec | None = None
    skills: SkillsSpec | None = None
    languages: LanguagesSpec | None = None
    min_years_exp: int | None = None
    temporality: Literal["current", "past", "any"] = "any"
    view_weights: dict[ViewName, float] | None = None


# ---------- Result structures ----------

class CandidateResult(BaseModel):
    candidate_id: str
    rank: int
    score: float
    match_explanation: str
    highlights: list[str] = Field(default_factory=list)
    per_dim: dict[str, float] | None = None  # populated only for det_picks


# ---------- Request / Response models ----------

class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None  # no-op; reserved for future use


class ChatResponse(BaseModel):
    query: str
    parsed_spec: ParsedSpec
    rag_picks: list[CandidateResult]
    det_picks: list[CandidateResult]
    suggested: list[CandidateResult]
    reasoning: str


class IngestRequest(BaseModel):
    force: bool = False  # re-embed even if the Chroma collection already exists


class IngestResponse(BaseModel):
    candidates_indexed: int
    documents_written: int
    duration_seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, bool]
```

- [ ] **Step 4: Run tests — expect pass**

```bash
poetry run pytest tests/test_models.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat(models): Pydantic models for parsed spec, results, API requests/responses"
```

---

### Task 3: DB connection + candidate loader

**Files:**
- Create: `app/db.py`
- Test: `tests/test_db.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write the conftest with a settings fixture**

Create `tests/conftest.py`:

```python
import pytest
from app.config import Settings


@pytest.fixture
def settings(monkeypatch, tmp_path) -> Settings:
    """Minimal valid settings for unit tests."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://developer:devread2024@34.79.32.228:5432/candidate_profiles")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))
    return Settings()


@pytest.fixture
def real_db_settings(settings):
    """Alias — use this fixture when the test hits the real DB (read-only)."""
    return settings
```

- [ ] **Step 2: Write DB tests first**

Create `tests/test_db.py`:

```python
import pytest
from app.db import fetch_all_candidates, fetch_candidate_bundle


@pytest.mark.integration
def test_fetch_all_candidates_returns_10k(real_db_settings):
    rows = fetch_all_candidates(real_db_settings.database_url)
    assert len(rows) == 10120
    assert "id" in rows[0]
    assert "first_name" in rows[0]
    assert "city" in rows[0]           # enriched
    assert "nationality" in rows[0]    # enriched


@pytest.mark.integration
def test_fetch_candidate_bundle_for_known_candidate(real_db_settings):
    # Sara Ali (from exploration)
    cid = "70222c8e-2b7a-4a9e-bc42-9ae3eaa2a89a"
    bundle = fetch_candidate_bundle(real_db_settings.database_url, cid)

    assert bundle["candidate"]["first_name"] == "Sara"
    assert bundle["candidate"]["last_name"] == "Ali"
    assert len(bundle["work"]) >= 1
    assert len(bundle["education"]) >= 1
    assert len(bundle["skills"]) >= 1
```

- [ ] **Step 3: Run — expect failure**

```bash
poetry run pytest tests/test_db.py -v
```

Expected: ImportError / ModuleNotFoundError.

- [ ] **Step 4: Implement `app/db.py`**

Leverage existing queries from `profile.py` and `fetch_profiles.py`. Factor into a single module that other services import.

```python
"""Postgres access — candidate data loaders."""
from __future__ import annotations

import psycopg2
from psycopg2.extras import RealDictCursor


ENRICHED_CANDIDATES_SQL = """
SELECT
    c.id, c.first_name, c.last_name, c.email, c.phone,
    c.date_of_birth, c.gender, c.headline, c.years_of_experience, c.created_at,
    city.name         AS city,
    city_country.name AS country,
    city_country.code AS country_code,
    nat.name          AS nationality,
    nat.code          AS nationality_code
FROM candidates c
LEFT JOIN cities    city         ON city.id = c.city_id
LEFT JOIN countries city_country ON city_country.id = city.country_id
LEFT JOIN countries nat          ON nat.id = c.nationality_id
ORDER BY c.id
"""

CANDIDATE_BY_ID_SQL = ENRICHED_CANDIDATES_SQL.replace("ORDER BY c.id", "WHERE c.id = %s")

WORK_SQL = """
SELECT we.job_title, we.start_date, we.end_date, we.is_current, we.description,
       co.name AS company, co.industry, co_country.name AS company_country,
       co_country.code AS company_country_code
FROM work_experience we
LEFT JOIN companies co         ON co.id = we.company_id
LEFT JOIN countries co_country ON co_country.id = co.country_id
WHERE we.candidate_id = %s
ORDER BY we.is_current DESC, we.start_date DESC
"""

EDUCATION_SQL = """
SELECT e.start_year, e.graduation_year, e.grade,
       i.name AS institution, d.name AS degree, f.name AS field
FROM education e
LEFT JOIN institutions    i ON i.id = e.institution_id
LEFT JOIN degrees         d ON d.id = e.degree_id
LEFT JOIN fields_of_study f ON f.id = e.field_of_study_id
WHERE e.candidate_id = %s
ORDER BY e.graduation_year DESC NULLS LAST
"""

SKILLS_SQL = """
SELECT s.name AS skill, sc.name AS category,
       cs.years_of_experience, cs.proficiency_level
FROM candidate_skills cs
JOIN skills           s  ON s.id = cs.skill_id
LEFT JOIN skill_categories sc ON sc.id = s.category_id
WHERE cs.candidate_id = %s
ORDER BY cs.years_of_experience DESC NULLS LAST, s.name
"""

LANGUAGES_SQL = """
SELECT l.name AS language, pl.name AS proficiency, pl.rank
FROM candidate_languages cl
JOIN languages          l  ON l.id = cl.language_id
JOIN proficiency_levels pl ON pl.id = cl.proficiency_level_id
WHERE cl.candidate_id = %s
ORDER BY pl.rank DESC, l.name
"""


def fetch_all_candidates(dsn: str) -> list[dict]:
    """Return one enriched row per candidate (10k rows). Used by the offline indexer."""
    with psycopg2.connect(dsn) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(ENRICHED_CANDIDATES_SQL)
            return [dict(r) for r in cur.fetchall()]


def _run(cur, sql, params):
    cur.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]


def fetch_candidate_bundle(dsn: str, candidate_id: str) -> dict:
    """Return everything we know about one candidate — candidate row + work + edu + skills + languages."""
    with psycopg2.connect(dsn) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(CANDIDATE_BY_ID_SQL, (candidate_id,))
            cand = cur.fetchone()
            if cand is None:
                raise LookupError(f"candidate not found: {candidate_id}")
            return {
                "candidate": dict(cand),
                "work":      _run(cur, WORK_SQL,      (candidate_id,)),
                "education": _run(cur, EDUCATION_SQL, (candidate_id,)),
                "skills":    _run(cur, SKILLS_SQL,    (candidate_id,)),
                "languages": _run(cur, LANGUAGES_SQL, (candidate_id,)),
            }


def fetch_all_bundles(dsn: str) -> list[dict]:
    """All 10k candidates with full bundles. Used by the offline indexer.

    Streams one connection; N+1 queries per candidate but runs once offline.
    Total ~50k queries over a persistent connection: ~1-2 minutes.
    """
    candidates = fetch_all_candidates(dsn)
    results: list[dict] = []
    with psycopg2.connect(dsn) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for c in candidates:
                cid = str(c["id"])
                results.append({
                    "candidate": c,
                    "work":      _run(cur, WORK_SQL,      (cid,)),
                    "education": _run(cur, EDUCATION_SQL, (cid,)),
                    "skills":    _run(cur, SKILLS_SQL,    (cid,)),
                    "languages": _run(cur, LANGUAGES_SQL, (cid,)),
                })
    return results
```

- [ ] **Step 5: Run tests — expect pass**

```bash
poetry run pytest tests/test_db.py -v -m integration
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add app/db.py tests/test_db.py tests/conftest.py
git commit -m "feat(db): candidate loaders — fetch_all_candidates, fetch_candidate_bundle, fetch_all_bundles"
```

---

## Phase 1 — Offline indexing

### Task 4: Probe text generation (3 views per candidate)

**Files:**
- Create: `app/probe_texts.py`
- Test: `tests/test_probe_texts.py`
- Modify: `tests/conftest.py` (add `sample_bundle` fixture)

- [ ] **Step 1: Add `sample_bundle` fixture to conftest**

Append to `tests/conftest.py`:

```python
import datetime as dt


@pytest.fixture
def sample_bundle():
    """Hardcoded Sara-Ali-like bundle for unit tests (no DB required)."""
    return {
        "candidate": {
            "id": "cid-001",
            "first_name": "Sara",
            "last_name": "Ali",
            "email": "saraali@hotmail.com",
            "phone": "+97323374090",
            "date_of_birth": dt.date(1980, 1, 4),
            "gender": "Female",
            "headline": "Results-driven Engineering Manager with expertise in Brand Marketing",
            "years_of_experience": 16,
            "city": "Philadelphia",
            "country": "United States",
            "country_code": "US",
            "nationality": "Latvia",
            "nationality_code": "LV",
            "created_at": dt.datetime(2026, 2, 20, 7, 38, 21),
        },
        "work": [
            {"job_title": "Engineering Manager Research Scientist",
             "start_date": dt.date(2025, 4, 1), "end_date": None,
             "is_current": True, "description": "Spearheaded migration to analysis, reducing costs by 58%.",
             "company": "IDM Brokerage House", "industry": "Finance",
             "company_country": "Czech Republic", "company_country_code": "CZ"},
        ],
        "education": [
            {"start_year": 2007, "graduation_year": 2010, "grade": "B-",
             "institution": "Union High School", "degree": "Associate's",
             "field": "Petroleum and Natural Gas Engineering"},
        ],
        "skills": [
            {"skill": "Brand Marketing", "category": "Marketing",
             "years_of_experience": 9, "proficiency_level": "Expert"},
            {"skill": "Analysis", "category": "Business",
             "years_of_experience": 6, "proficiency_level": "Expert"},
        ],
        "languages": [
            {"language": "Italian", "proficiency": "Intermediate", "rank": 2},
        ],
    }
```

- [ ] **Step 2: Write probe text tests**

Create `tests/test_probe_texts.py`:

```python
from app.probe_texts import render_views


def test_render_returns_three_views(sample_bundle):
    views = render_views(sample_bundle)
    assert set(views.keys()) == {"summary", "work", "skills_edu"}


def test_summary_mentions_name_title_location(sample_bundle):
    views = render_views(sample_bundle)
    s = views["summary"]
    assert "Sara Ali" in s
    assert "Engineering Manager" in s
    assert "Philadelphia" in s
    assert "Latvia" in s  # nationality
    assert "16 years" in s


def test_work_mentions_company_and_dates(sample_bundle):
    views = render_views(sample_bundle)
    w = views["work"]
    assert "IDM Brokerage House" in w
    assert "Finance" in w
    assert "2025" in w


def test_skills_edu_mentions_skill_and_degree(sample_bundle):
    views = render_views(sample_bundle)
    se = views["skills_edu"]
    assert "Brand Marketing" in se
    assert "9" in se         # years for brand marketing
    assert "Associate" in se  # degree


def test_empty_sections_do_not_crash(sample_bundle):
    sample_bundle["work"] = []
    sample_bundle["education"] = []
    sample_bundle["skills"] = []
    sample_bundle["languages"] = []
    views = render_views(sample_bundle)
    assert views["summary"]              # header-only summary still produced
    assert views["work"] == "" or "no work history" in views["work"].lower()
```

- [ ] **Step 3: Run — expect failure**

```bash
poetry run pytest tests/test_probe_texts.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `app/probe_texts.py`**

```python
"""Generate 3 natural-language probe texts per candidate for vector/BM25 indexing."""
from __future__ import annotations


def _full_name(c: dict) -> str:
    return f"{c['first_name']} {c['last_name']}"


def _location(c: dict) -> str:
    parts = [p for p in (c.get("city"), c.get("country")) if p]
    return ", ".join(parts) if parts else "location unknown"


def render_summary(bundle: dict) -> str:
    c = bundle["candidate"]
    name = _full_name(c)
    headline = (c.get("headline") or "").strip()
    yrs = c.get("years_of_experience")
    location = _location(c)
    nationality = c.get("nationality") or "unknown nationality"

    current = next((w for w in bundle.get("work", []) if w.get("is_current")), None)
    current_sentence = (
        f"Currently {current['job_title']} at {current['company']}"
        + (f" ({current['industry']})" if current.get("industry") else "")
        + (f" in {current['company_country']}" if current.get("company_country") else "")
        + "."
    ) if current else ""

    parts = [
        f"{name} is a professional with {yrs} years of experience." if yrs is not None else f"{name}.",
    ]
    if headline:
        parts.append(headline + ".")
    if current_sentence:
        parts.append(current_sentence)
    parts.append(f"Based in {location}; {nationality} nationality.")
    return " ".join(parts)


def render_work(bundle: dict) -> str:
    jobs = bundle.get("work", [])
    if not jobs:
        return f"{_full_name(bundle['candidate'])} has no work history on record."

    name = _full_name(bundle["candidate"])
    lines = [f"{name}'s career includes:"]
    for j in jobs:
        period = (
            f"{j['start_date'].year}"
            + (f"–{j['end_date'].year}" if j.get("end_date") else "–present")
        )
        industry = f", {j['industry']}" if j.get("industry") else ""
        country = f", {j['company_country']}" if j.get("company_country") else ""
        desc = f" {j['description']}" if j.get("description") else ""
        lines.append(f"- {j['job_title']} at {j['company']}{industry}{country} ({period}).{desc}")
    return "\n".join(lines)


def render_skills_edu(bundle: dict) -> str:
    name = _full_name(bundle["candidate"])
    skills = bundle.get("skills", [])
    edu = bundle.get("education", [])
    langs = bundle.get("languages", [])

    parts: list[str] = []
    if skills:
        top = sorted(skills, key=lambda s: -(s.get("years_of_experience") or 0))[:8]
        skill_strs = [
            f"{s['skill']} ({s['years_of_experience']}y, {s.get('proficiency_level', 'n/a')})"
            for s in top
        ]
        parts.append(f"{name}'s top skills: {', '.join(skill_strs)}.")
    if edu:
        edu_strs = []
        for e in edu:
            edu_strs.append(
                f"{e.get('degree','Degree')} in {e.get('field','—')} "
                f"from {e.get('institution','—')} "
                f"({e.get('start_year','?')}–{e.get('graduation_year','?')})"
            )
        parts.append("Education: " + "; ".join(edu_strs) + ".")
    if langs:
        lang_strs = [f"{l['language']} ({l['proficiency']})" for l in langs]
        parts.append("Languages: " + ", ".join(lang_strs) + ".")
    return " ".join(parts) if parts else f"{name} has no skills/education/languages on record."


def render_views(bundle: dict) -> dict[str, str]:
    """Return the three probe texts for a candidate bundle."""
    return {
        "summary":    render_summary(bundle),
        "work":       render_work(bundle),
        "skills_edu": render_skills_edu(bundle),
    }
```

- [ ] **Step 5: Run tests — expect pass**

```bash
poetry run pytest tests/test_probe_texts.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add app/probe_texts.py tests/test_probe_texts.py tests/conftest.py
git commit -m "feat(probe_texts): render summary/work/skills_edu probe texts per candidate"
```

---

### Task 5: Embedding wrapper

**Files:**
- Create: `app/embeddings.py`
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Write tests first (with mocking)**

Create `tests/test_embeddings.py`:

```python
from unittest.mock import MagicMock, patch

from app.embeddings import EmbeddingClient


def test_embed_single_returns_float_vector():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )
    with patch("app.embeddings.OpenAI", return_value=mock_client):
        c = EmbeddingClient(api_key="x", model="any/model")
        vec = c.embed_one("hello")
    assert vec == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once()


def test_embed_batch_returns_list_of_vectors():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1]*3), MagicMock(embedding=[0.2]*3)]
    )
    with patch("app.embeddings.OpenAI", return_value=mock_client):
        c = EmbeddingClient(api_key="x", model="any/model")
        vecs = c.embed_batch(["a", "b"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 3
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_embeddings.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/embeddings.py`**

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
poetry run pytest tests/test_embeddings.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/embeddings.py tests/test_embeddings.py
git commit -m "feat(embeddings): OpenRouter-compatible embedding client with retry"
```

---

### Task 6: Chroma store + BM25 index

**Files:**
- Create: `app/chroma_store.py`
- Create: `app/bm25_index.py`
- Test: `tests/test_chroma_store.py`
- Test: `tests/test_bm25_index.py`

- [ ] **Step 1: Write Chroma store tests**

Create `tests/test_chroma_store.py`:

```python
import numpy as np
import pytest

from app.chroma_store import ChromaStore


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_path=str(tmp_path / "chroma"))


def test_upsert_and_load_roundtrip(store):
    store.upsert_batch(
        ids=["c-1::summary", "c-1::work", "c-2::summary"],
        embeddings=[[0.1]*3, [0.2]*3, [0.3]*3],
        documents=["sum1", "work1", "sum2"],
        metadatas=[
            {"candidate_id": "c-1", "view": "summary"},
            {"candidate_id": "c-1", "view": "work"},
            {"candidate_id": "c-2", "view": "summary"},
        ],
    )
    embs, meta, docs, ids = store.load_all()
    assert embs.shape == (3, 3)
    assert len(ids) == 3
    assert all(m["view"] in {"summary", "work"} for m in meta)


def test_count_after_upsert(store):
    assert store.count() == 0
    store.upsert_batch(
        ids=["c-1::summary"],
        embeddings=[[0.5]*3],
        documents=["x"],
        metadatas=[{"candidate_id": "c-1", "view": "summary"}],
    )
    assert store.count() == 1
```

- [ ] **Step 2: Write BM25 tests**

Create `tests/test_bm25_index.py`:

```python
from app.bm25_index import BM25Index


def test_bm25_scores_for_matching_query():
    docs = ["python expert regulatory affairs", "sales manager pharma", "pharma regulatory"]
    idx = BM25Index.build(docs)
    scores = idx.score("regulatory pharma")
    assert len(scores) == 3
    # doc 2 (pharma regulatory) should score highest
    assert scores[2] > scores[0]
    assert scores[2] > scores[1]


def test_bm25_tokenizer_lowercase():
    docs = ["Python EXPERT"]
    idx = BM25Index.build(docs)
    assert idx.score("python")[0] > 0
    assert idx.score("PYTHON")[0] > 0
```

- [ ] **Step 3: Run — expect failure**

```bash
poetry run pytest tests/test_chroma_store.py tests/test_bm25_index.py -v
```

Expected: ImportError on both.

- [ ] **Step 4: Implement `app/chroma_store.py`**

```python
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
```

- [ ] **Step 5: Implement `app/bm25_index.py`**

```python
"""rank_bm25 wrapper with simple whitespace + lowercase tokenization."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi
import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class BM25Index:
    def __init__(self, bm25: BM25Okapi):
        self._bm25 = bm25

    @classmethod
    def build(cls, documents: list[str]) -> "BM25Index":
        tokenized = [tokenize(d) for d in documents]
        # rank_bm25 crashes on empty corpora; guard
        if not tokenized:
            tokenized = [[""]]
        return cls(BM25Okapi(tokenized))

    def score(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
```

- [ ] **Step 6: Run tests — expect pass**

```bash
poetry run pytest tests/test_chroma_store.py tests/test_bm25_index.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add app/chroma_store.py app/bm25_index.py tests/test_chroma_store.py tests/test_bm25_index.py
git commit -m "feat(storage): Chroma store wrapper + BM25 index wrapper"
```

---

### Task 7: Offline indexer

**Files:**
- Create: `app/ingest.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write indexer tests (mocking embeddings + DB)**

Create `tests/test_ingest.py`:

```python
from unittest.mock import MagicMock, patch

from app.ingest import build_index
from app.chroma_store import ChromaStore


def test_build_index_upserts_three_docs_per_candidate(tmp_path, sample_bundle):
    store = ChromaStore(persist_path=str(tmp_path / "chroma"))

    with patch("app.ingest.fetch_all_bundles", return_value=[sample_bundle]):
        with patch("app.ingest.EmbeddingClient") as EmbClient:
            mock_emb = EmbClient.return_value
            mock_emb.embed_batch.return_value = [[0.1, 0.2, 0.3]] * 3

            result = build_index(
                dsn="postgresql://fake",
                api_key="x",
                embedding_model="m",
                store=store,
            )

    assert result["candidates_indexed"] == 1
    assert result["documents_written"] == 3
    assert store.count() == 3

    embs, meta, docs, ids = store.load_all()
    views = {m["view"] for m in meta}
    assert views == {"summary", "work", "skills_edu"}
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_ingest.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/ingest.py`**

```python
"""Offline indexing: load candidates → render probe texts → embed → write to Chroma."""
from __future__ import annotations

import time

from app.chroma_store import ChromaStore
from app.db import fetch_all_bundles
from app.embeddings import EmbeddingClient
from app.probe_texts import render_views


def build_index(
    dsn: str,
    api_key: str,
    embedding_model: str,
    store: ChromaStore,
    batch_size: int = 64,
) -> dict:
    """Build the Chroma index. Returns a summary dict."""
    t0 = time.monotonic()
    store.reset()

    bundles = fetch_all_bundles(dsn)
    embedder = EmbeddingClient(api_key=api_key, model=embedding_model)

    # Stage all (text, id, metadata) triples
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for b in bundles:
        cid = str(b["candidate"]["id"])
        views = render_views(b)
        for view_name, text in views.items():
            ids.append(f"{cid}::{view_name}")
            texts.append(text)
            metadatas.append({"candidate_id": cid, "view": view_name})

    # Embed in batches and upsert
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        embs = embedder.embed_batch(batch_texts)
        store.upsert_batch(
            ids=ids[i : i + batch_size],
            embeddings=embs,
            documents=batch_texts,
            metadatas=metadatas[i : i + batch_size],
        )

    return {
        "candidates_indexed": len(bundles),
        "documents_written":  len(texts),
        "duration_seconds":   round(time.monotonic() - t0, 2),
    }
```

- [ ] **Step 4: Run tests — expect pass**

```bash
poetry run pytest tests/test_ingest.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add app/ingest.py tests/test_ingest.py
git commit -m "feat(ingest): end-to-end offline indexer (load → render → embed → upsert)"
```

---

## Phase 2 — LLM infrastructure

### Task 8: LLM wrapper

**Files:**
- Create: `app/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write tests**

Create `tests/test_llm.py`:

```python
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
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_llm.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/llm.py`**

```python
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
    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_json(self, messages: list[dict], temperature: float = 0.0) -> dict:
        """Call chat and parse the content as JSON; tolerate ```json fences."""
        raw = self.chat(messages, temperature=temperature)
        m = _FENCE_RE.match(raw)
        text = m.group(1) if m else raw
        return json.loads(text)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
poetry run pytest tests/test_llm.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/llm.py tests/test_llm.py
git commit -m "feat(llm): OpenRouter chat client with JSON helper + retry"
```

---

### Task 9: Query parser

**Files:**
- Create: `app/query_parser.py`
- Test: `tests/test_query_parser.py`

- [ ] **Step 1: Write parser tests (mocked LLM)**

Create `tests/test_query_parser.py`:

```python
from unittest.mock import MagicMock

from app.query_parser import parse_query
from app.models import ParsedSpec


def test_parse_pharma_middle_east_query():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "function":  {"values": ["Regulatory Affairs"], "weight": 0.35, "required": True},
        "industry":  {"values": ["Pharmaceuticals"],    "weight": 0.30, "required": True},
        "geography": {"values": ["AE","SA"], "weight": 0.20, "required": False,
                      "location_type": "current_or_nationality"},
        "temporality": "any",
        "view_weights": {"summary": 0.3, "work": 0.5, "skills_edu": 0.2},
    }

    spec = parse_query("regulatory affairs experts in pharma in the Middle East", llm=llm)

    assert isinstance(spec, ParsedSpec)
    assert spec.function.required is True
    assert spec.industry.values == ["Pharmaceuticals"]
    assert "AE" in spec.geography.values
    assert spec.view_weights["work"] == 0.5


def test_parse_returns_temporality_default_if_missing():
    llm = MagicMock()
    llm.chat_json.return_value = {}  # totally empty
    spec = parse_query("find anyone", llm=llm)
    assert spec.temporality == "any"
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_query_parser.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/query_parser.py`**

```python
"""NL query → ParsedSpec via a single LLM call."""
from __future__ import annotations

from app.llm import LLMClient
from app.models import ParsedSpec


SYSTEM_PROMPT = """You are a candidate-search query parser. Convert the user's natural-language
query into a structured JSON spec. Your output MUST be valid JSON matching this schema:

{
  "function":   {"values": [str], "weight": float (0-1), "required": bool} | null,
  "industry":   {"values": [str], "weight": float (0-1), "required": bool} | null,
  "geography":  {"values": [str (ISO-2 country codes)], "weight": float, "required": bool,
                 "location_type": "current" | "current_or_nationality" | "historical"} | null,
  "seniority":  {"levels": ["junior"|"mid"|"senior"|"executive"], "weight": float, "required": bool} | null,
  "skills":     {"values": [str], "weight": float, "required": bool} | null,
  "languages":  {"values": [str], "required_proficiency": str | null, "weight": float, "required": bool} | null,
  "min_years_exp": int | null,
  "temporality": "current" | "past" | "any",
  "view_weights": {"summary": float, "work": float, "skills_edu": float} | null
}

Rules:
- EXPAND entities: "Middle East" -> ISO codes ["AE","SA","QA","BH","KW","OM","EG","JO","LB"]; "GCC" -> ["AE","SA","QA","BH","KW","OM"]; "pharma" -> ["Pharmaceuticals","Biotechnology"].
- FIX typos silently (e.g., "pharmacuetical" -> "Pharmaceutical").
- Use `required: true` ONLY when the query uses "must", "only", "specifically"; otherwise `false`.
- Weights must sum to <= 1.0 across the dimensions you populate.
- `view_weights` values sum to 1.0; pick them based on what the query emphasizes:
  - work-history-heavy query ("former CPO", "worked at") -> work=0.5-0.6
  - skill-heavy query ("Python expert", "knows Kubernetes") -> skills_edu=0.5
  - generic role search -> roughly balanced, summary slightly higher
- "former X" implies temporality="past"; "currently X" implies "current"; otherwise "any".

CUSTOM WEIGHTING HEURISTICS (user-supplied):
# TODO: user writes 5-10 if-then rules here based on their domain knowledge. Example:
# - If query mentions "former [role]": temporality=past, seniority.weight >= 0.3
# - If query mentions "junior": seniority=["junior"], geography.required=false
# - If query specifies a company name: treat it as a hard filter

Output ONLY the JSON object. No prose, no markdown fences."""


def parse_query(query: str, llm: LLMClient) -> ParsedSpec:
    obj = llm.chat_json([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ])
    return ParsedSpec.model_validate(obj)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
poetry run pytest tests/test_query_parser.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/query_parser.py tests/test_query_parser.py
git commit -m "feat(query_parser): LLM-based NL->ParsedSpec with entity expansion"
```

---

### Task 10: HyDE

**Files:**
- Create: `app/hyde.py`
- Test: `tests/test_hyde.py`

- [ ] **Step 1: Write tests**

Create `tests/test_hyde.py`:

```python
from unittest.mock import MagicMock

from app.hyde import generate_hypothetical_profile


def test_hyde_returns_text():
    llm = MagicMock()
    llm.chat.return_value = "Jane Doe is a Regulatory Affairs Director with 12 years in pharma..."
    text = generate_hypothetical_profile("regulatory affairs in pharma in ME", llm=llm)
    assert "Regulatory" in text
    assert len(text) > 20
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_hyde.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/hyde.py`**

```python
"""Hypothetical Document Embeddings: LLM writes an ideal candidate profile for the query."""
from __future__ import annotations

from app.llm import LLMClient


SYSTEM_PROMPT = """You write short, natural-language candidate profiles.

Given a search query, invent a hypothetical ideal candidate profile that would perfectly
match the query. Include: name, current role & company, years of experience, location,
2-3 relevant past roles, and top skills. 3-5 sentences. Plain prose, no headers, no bullets.
This text will be embedded and used as a retrieval query, so make it concrete and realistic."""


def generate_hypothetical_profile(query: str, llm: LLMClient) -> str:
    return llm.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Query: {query}\n\nHypothetical candidate profile:"},
        ],
        temperature=0.4,
        max_tokens=300,
    ).strip()
```

- [ ] **Step 4: Run tests — expect pass**

```bash
poetry run pytest tests/test_hyde.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add app/hyde.py tests/test_hyde.py
git commit -m "feat(hyde): hypothetical profile generator for retrieval query"
```

---

## Phase 3 — RAG Agent

### Task 11: RAG retrieval (vector + BM25 + RRF)

**Files:**
- Create: `app/rag_agent.py` (retrieval half only)
- Test: `tests/test_rag_agent.py`

- [ ] **Step 1: Write retrieval tests**

Create `tests/test_rag_agent.py`:

```python
import numpy as np

from app.rag_agent import (
    aggregate_per_candidate,
    rrf_fuse,
    retrieve_candidates,
)
from app.models import ParsedSpec


def test_aggregate_weighted_sum_when_view_weights_present():
    sims = np.array([
        [0.8, 0.6, 0.4],  # candidate 0
        [0.5, 0.5, 0.5],  # candidate 1
    ])
    weights = {"summary": 0.5, "work": 0.3, "skills_edu": 0.2}
    agg = aggregate_per_candidate(sims, weights)
    # c0: 0.5*0.8 + 0.3*0.6 + 0.2*0.4 = 0.66
    # c1: 0.5*0.5 + 0.3*0.5 + 0.2*0.5 = 0.50
    assert abs(agg[0] - 0.66) < 1e-6
    assert abs(agg[1] - 0.50) < 1e-6


def test_aggregate_falls_back_to_max_when_weights_none():
    sims = np.array([[0.8, 0.6, 0.4], [0.5, 0.9, 0.1]])
    agg = aggregate_per_candidate(sims, None)
    assert abs(agg[0] - 0.8) < 1e-6
    assert abs(agg[1] - 0.9) < 1e-6


def test_aggregate_falls_back_to_max_when_weights_all_zero():
    sims = np.array([[0.8, 0.6, 0.4]])
    agg = aggregate_per_candidate(sims, {"summary": 0, "work": 0, "skills_edu": 0})
    assert abs(agg[0] - 0.8) < 1e-6


def test_rrf_fuse_combines_ranks():
    vec_scores  = np.array([0.9, 0.1, 0.5])
    bm25_scores = np.array([0.1, 0.9, 0.5])
    fused = rrf_fuse(vec_scores, bm25_scores, k=60)
    # Candidate 2 is rank 2 in both -> highest fused
    assert np.argmax(fused) == 2


def test_retrieve_candidates_returns_top_k():
    # 3 candidates, 3 views, embedding dim 4
    all_embs = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],  # candidate 0: summary, work, skills
        [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],  # candidate 1
        [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],  # candidate 2 (orthogonal)
    ], dtype=np.float32)
    candidate_ids = ["c0", "c0", "c0", "c1", "c1", "c1", "c2", "c2", "c2"]
    documents    = ["c0sum", "c0work", "c0skills"] * 3  # placeholder docs

    query_vec = np.array([1, 0, 0, 0], dtype=np.float32)

    results = retrieve_candidates(
        all_embs=all_embs,
        candidate_ids=candidate_ids,
        documents=documents,
        query_vec=query_vec,
        query_text="summary",
        view_weights=None,
        top_k=2,
    )
    assert len(results) == 2
    assert results[0]["candidate_id"] == "c0"   # best match on summary view
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_rag_agent.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement retrieval half of `app/rag_agent.py`**

```python
"""RAG agent — full-scan hybrid retrieval with RRF, top-K candidates."""
from __future__ import annotations

import numpy as np

from app.bm25_index import BM25Index


VIEW_ORDER = ("summary", "work", "skills_edu")


def aggregate_per_candidate(sims_matrix: np.ndarray, view_weights: dict | None) -> np.ndarray:
    """sims_matrix shape: (N_candidates, 3). Returns (N_candidates,).

    Uses query-weighted sum when view_weights has any positive value;
    falls back to max otherwise.
    """
    if view_weights and any(view_weights.get(v, 0) > 0 for v in VIEW_ORDER):
        w = np.array([view_weights.get(v, 0.0) for v in VIEW_ORDER], dtype=np.float32)
        return sims_matrix @ w
    return sims_matrix.max(axis=1)


def rrf_fuse(vec_scores: np.ndarray, bm25_scores: np.ndarray, k: int = 60) -> np.ndarray:
    """Combine two per-candidate score arrays via Reciprocal Rank Fusion."""
    # argsort gives ascending order; for descending rank take the inverse index
    def _ranks_desc(a: np.ndarray) -> np.ndarray:
        order = np.argsort(-a)                # indices sorted best-first
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(a))      # rank 0 for best, 1 for second, ...
        return ranks
    rv = _ranks_desc(vec_scores)
    rb = _ranks_desc(bm25_scores)
    return 1.0 / (k + rv) + 1.0 / (k + rb)


def _group_by_candidate(
    all_embs: np.ndarray, candidate_ids: list[str], documents: list[str], views: list[str] | None = None
) -> tuple[np.ndarray, list[str], list[list[str]]]:
    """Return (sims_matrix, unique_candidate_ids, docs_per_candidate).

    Assumes embeddings are already ordered (candidate, view) with exactly 3 views per candidate
    in VIEW_ORDER. `views` is the optional parallel view name list (for assertion).
    """
    assert len(all_embs) == len(candidate_ids) == len(documents)
    unique_ids: list[str] = []
    seen: set[str] = set()
    for cid in candidate_ids:
        if cid not in seen:
            unique_ids.append(cid)
            seen.add(cid)
    n = len(unique_ids)
    return unique_ids


def retrieve_candidates(
    all_embs: np.ndarray,
    candidate_ids: list[str],
    documents: list[str],
    query_vec: np.ndarray,
    query_text: str,
    view_weights: dict | None,
    top_k: int = 50,
    bm25_index: BM25Index | None = None,
) -> list[dict]:
    """Full-scan hybrid retrieval. Returns top_k candidate dicts sorted by fused score.

    Each dict: {candidate_id, score, rank, views: {view_name: (doc_text, vec_sim, bm25_score)}}
    """
    # 1. Cosine similarity (assumes embeddings are L2-normalized; if not, normalize here)
    # For defensive behavior we just use raw dot product.
    vec_sims_flat = (all_embs @ query_vec).astype(np.float32)

    # 2. BM25 scores for every document
    if bm25_index is None:
        bm25_index = BM25Index.build(documents)
    bm25_flat = bm25_index.score(query_text)

    # 3. Group by candidate: we expect (c, v) ordering with 3 views per candidate, in VIEW_ORDER
    unique_ids: list[str] = []
    seen: set[str] = set()
    for cid in candidate_ids:
        if cid not in seen:
            unique_ids.append(cid)
            seen.add(cid)
    n_candidates = len(unique_ids)
    assert len(candidate_ids) == n_candidates * 3, "Expect exactly 3 views per candidate"

    vec_mat  = vec_sims_flat.reshape(n_candidates, 3)
    bm25_mat = bm25_flat.reshape(n_candidates, 3)

    agg_vec  = aggregate_per_candidate(vec_mat,  view_weights)
    agg_bm25 = aggregate_per_candidate(bm25_mat, view_weights)

    fused = rrf_fuse(agg_vec, agg_bm25)

    top_idx = np.argsort(-fused)[:top_k]
    results: list[dict] = []
    for rank, idx in enumerate(top_idx, start=1):
        base = idx * 3
        results.append({
            "candidate_id": unique_ids[idx],
            "score": float(fused[idx]),
            "rank": rank,
            "documents": {
                VIEW_ORDER[v]: documents[base + v] for v in range(3)
            },
            "vec_per_view":  {VIEW_ORDER[v]: float(vec_mat[idx, v])  for v in range(3)},
            "bm25_per_view": {VIEW_ORDER[v]: float(bm25_mat[idx, v]) for v in range(3)},
        })
    return results
```

- [ ] **Step 4: Run tests — expect pass**

```bash
poetry run pytest tests/test_rag_agent.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/rag_agent.py tests/test_rag_agent.py
git commit -m "feat(rag): full-scan hybrid retrieval with aggregation + RRF fusion"
```

---

### Task 12: Listwise LLM rerank

**Files:**
- Modify: `app/rag_agent.py` (add `rerank_and_explain`)
- Modify: `tests/test_rag_agent.py` (add rerank tests)

- [ ] **Step 1: Write rerank tests**

Append to `tests/test_rag_agent.py`:

```python
from unittest.mock import MagicMock
from app.rag_agent import rerank_and_explain


def test_rerank_returns_5_candidate_results():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "picks": [
            {"candidate_id": "c1", "match_explanation": "best semantic fit",
             "highlights": ["12y pharma", "Dubai"]},
            {"candidate_id": "c3", "match_explanation": "solid run",
             "highlights": ["10y pharma"]},
            {"candidate_id": "c5", "match_explanation": "good signals",
             "highlights": ["regulatory lead"]},
            {"candidate_id": "c2", "match_explanation": "ok",
             "highlights": ["9y pharma"]},
            {"candidate_id": "c4", "match_explanation": "decent",
             "highlights": ["8y pharma"]},
        ]
    }
    candidates = [
        {"candidate_id": f"c{i}", "score": 0.1*i, "rank": i,
         "documents": {"summary": f"profile-{i}", "work": f"work-{i}", "skills_edu": f"se-{i}"}}
        for i in range(1, 6)
    ]
    picks = rerank_and_explain(query="pharma ME regulatory", candidates=candidates, llm=llm)
    assert len(picks) == 5
    assert picks[0].candidate_id == "c1"
    assert picks[0].rank == 1
    assert "Dubai" in picks[0].highlights
    assert picks[0].match_explanation.startswith("best")
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_rag_agent.py::test_rerank_returns_5_candidate_results -v
```

Expected: ImportError on `rerank_and_explain`.

- [ ] **Step 3: Add `rerank_and_explain` to `app/rag_agent.py`**

Append to `app/rag_agent.py`:

```python
from app.llm import LLMClient
from app.models import CandidateResult


RERANK_SYSTEM_PROMPT = """You are a senior technical recruiter. Given a user query and a
list of candidate mini-profiles, pick the 5 best matches and explain why.

Output MUST be valid JSON of exactly this shape:
{
  "picks": [
    {
      "candidate_id": "<id from the input>",
      "match_explanation": "<one concise sentence on why this candidate ranks here>",
      "highlights": ["<2-4 short bullets of proof>", ...]
    },
    ...  (exactly 5 items)
  ]
}

Rules:
- The 5 picks must be the 5 strongest overall matches — make relative judgments.
- Each highlight must be concrete (e.g. "12 years at Pfizer" not "lots of experience").
- Output JSON only, no prose or markdown fences."""


def _format_candidate_block(c: dict) -> str:
    docs = c["documents"]
    return (
        f"[candidate_id: {c['candidate_id']}]\n"
        f"Summary: {docs['summary']}\n"
        f"Work:    {docs['work']}\n"
        f"Skills:  {docs['skills_edu']}\n"
    )


def rerank_and_explain(query: str, candidates: list[dict], llm: LLMClient) -> list[CandidateResult]:
    blocks = "\n\n".join(_format_candidate_block(c) for c in candidates)
    user = f"Query:\n{query}\n\nCandidates:\n{blocks}\n\nReturn the top-5 JSON."

    obj = llm.chat_json(
        [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ],
        temperature=0.0,
    )

    # Index original candidates by id to preserve scores
    by_id = {c["candidate_id"]: c for c in candidates}
    out: list[CandidateResult] = []
    for rank, pick in enumerate(obj["picks"][:5], start=1):
        cid = pick["candidate_id"]
        base = by_id.get(cid)
        score = float(base["score"]) if base else 0.0
        out.append(CandidateResult(
            candidate_id=cid,
            rank=rank,
            score=score,
            match_explanation=pick["match_explanation"],
            highlights=list(pick.get("highlights", [])),
        ))
    return out
```

- [ ] **Step 4: Run tests — expect pass**

```bash
poetry run pytest tests/test_rag_agent.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add app/rag_agent.py tests/test_rag_agent.py
git commit -m "feat(rag): listwise LLM rerank producing CandidateResult with highlights"
```

---

## Phase 4 — Deterministic agent

### Task 13: Scoring dimensions

**Files:**
- Create: `app/scoring.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write scoring dimension tests**

Create `tests/test_scoring.py`:

```python
import datetime as dt
import pytest

from app.models import (
    DimensionSpec, GeoSpec, SenioritySpec, SkillsSpec, LanguagesSpec,
)
from app.scoring import (
    score_function, score_industry, score_geography,
    score_seniority, score_skills, score_languages,
)


@pytest.fixture
def bundle_pharma_regulatory_uae():
    return {
        "candidate": {
            "id": "c-1", "first_name": "Ahmed", "last_name": "Hassan",
            "years_of_experience": 12,
            "headline": "Regulatory Affairs Director with deep pharma experience",
            "country_code": "AE", "nationality_code": "AE",
        },
        "work": [
            {"job_title": "Regulatory Affairs Director",
             "start_date": dt.date(2020, 1, 1), "end_date": None, "is_current": True,
             "company": "Pfizer", "industry": "Pharmaceuticals",
             "company_country_code": "AE"},
            {"job_title": "Senior Regulatory Associate",
             "start_date": dt.date(2015, 1, 1), "end_date": dt.date(2020, 1, 1), "is_current": False,
             "company": "Novartis", "industry": "Pharmaceuticals",
             "company_country_code": "CH"},
        ],
        "skills": [
            {"skill": "Regulatory Compliance", "years_of_experience": 10, "proficiency_level": "Expert"},
            {"skill": "FDA Submissions",       "years_of_experience": 7,  "proficiency_level": "Expert"},
        ],
        "languages": [
            {"language": "Arabic",  "proficiency": "Native",   "rank": 4},
            {"language": "English", "proficiency": "Fluent",   "rank": 3},
        ],
    }


def test_score_function_high_for_matching_title(bundle_pharma_regulatory_uae):
    spec = DimensionSpec(values=["Regulatory Affairs"], weight=0.35, required=True)
    score = score_function(bundle_pharma_regulatory_uae, spec)
    assert score > 0.7  # title is a near-exact match


def test_score_industry_full_credit_for_career_in_pharma(bundle_pharma_regulatory_uae):
    spec = DimensionSpec(values=["Pharmaceuticals"], weight=0.30, required=True)
    score = score_industry(bundle_pharma_regulatory_uae, spec)
    assert score == pytest.approx(1.0, abs=0.01)


def test_score_geography_current_country(bundle_pharma_regulatory_uae):
    spec = GeoSpec(values=["AE", "SA"], weight=0.20, required=False,
                   location_type="current_or_nationality")
    score = score_geography(bundle_pharma_regulatory_uae, spec)
    assert score == 1.0


def test_score_geography_returns_zero_if_no_match(bundle_pharma_regulatory_uae):
    spec = GeoSpec(values=["US"], weight=0.20, required=False,
                   location_type="current_or_nationality")
    assert score_geography(bundle_pharma_regulatory_uae, spec) == 0.0


def test_score_seniority_maps_12y_to_senior(bundle_pharma_regulatory_uae):
    spec = SenioritySpec(levels=["senior"], weight=0.10, required=False)
    assert score_seniority(bundle_pharma_regulatory_uae, spec) == 1.0


def test_score_seniority_adjacent_gets_half(bundle_pharma_regulatory_uae):
    spec = SenioritySpec(levels=["executive"], weight=0.10, required=False)
    # 12y maps to senior (adjacent to executive) -> 0.5
    assert score_seniority(bundle_pharma_regulatory_uae, spec) == 0.5


def test_score_skills_hits_over_target(bundle_pharma_regulatory_uae):
    spec = SkillsSpec(values=["Regulatory Compliance", "FDA Submissions", "Russian"],
                      weight=0.10, required=False)
    # 2 out of 3 matches, with years bonus
    score = score_skills(bundle_pharma_regulatory_uae, spec)
    assert 0.55 < score < 0.70


def test_score_languages_fraction_matched(bundle_pharma_regulatory_uae):
    spec = LanguagesSpec(values=["Arabic", "French"], required_proficiency=None,
                         weight=0.05, required=False)
    assert score_languages(bundle_pharma_regulatory_uae, spec) == 0.5
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_scoring.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/scoring.py`**

```python
"""Per-dimension scoring functions for the deterministic agent. Every function returns [0, 1]."""
from __future__ import annotations

import datetime as dt
from difflib import SequenceMatcher

from app.models import (
    DimensionSpec, GeoSpec, SenioritySpec, SkillsSpec, LanguagesSpec,
)


# ---------- Function ----------

SENIORITY_KEYWORDS = ("chief", "vp", "vice president", "head", "director", "principal", "lead")


def _trgm_sim(a: str, b: str) -> float:
    """Cheap trigram-like similarity using SequenceMatcher as a stand-in for pg_trgm."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def score_function(bundle: dict, spec: DimensionSpec) -> float:
    if not spec.values:
        return 0.0
    titles_with_recency: list[tuple[str, float]] = []
    today = dt.date.today()
    for w in bundle.get("work", []):
        title = w.get("job_title") or ""
        if w.get("is_current"):
            recency = 1.0
        else:
            end = w.get("end_date") or today
            years_since = (today - end).days / 365.25
            recency = max(0.3, 1.0 - 0.1 * years_since)  # decays, min 0.3
        titles_with_recency.append((title, recency))
    headline = bundle["candidate"].get("headline") or ""
    titles_with_recency.append((headline, 0.8))

    best = 0.0
    for target in spec.values:
        for title, rec in titles_with_recency:
            sim = _trgm_sim(target, title) * rec
            if sim > best:
                best = sim
    return min(best, 1.0)


# ---------- Industry ----------

def _years_of(w: dict) -> float:
    start = w.get("start_date")
    end = w.get("end_date") or dt.date.today()
    if not start:
        return 0.0
    return max(0.0, (end - start).days / 365.25)


def score_industry(bundle: dict, spec: DimensionSpec) -> float:
    if not spec.values:
        return 0.0
    targets = {t.lower() for t in spec.values}
    total = 0.0
    matched = 0.0
    for w in bundle.get("work", []):
        y = _years_of(w)
        total += y
        if (w.get("industry") or "").lower() in targets:
            matched += y
    if total <= 0:
        return 0.0
    return min(matched / total, 1.0)


# ---------- Geography ----------

def score_geography(bundle: dict, spec: GeoSpec) -> float:
    if not spec.values:
        return 0.0
    targets = {v.upper() for v in spec.values}
    c = bundle["candidate"]
    current = (c.get("country_code") or "").upper()
    nationality = (c.get("nationality_code") or "").upper()

    if spec.location_type == "current":
        return 1.0 if current in targets else 0.0
    if spec.location_type == "current_or_nationality":
        return 1.0 if (current in targets or nationality in targets) else 0.0
    # historical
    jobs = bundle.get("work", [])
    matches_any = any((w.get("company_country_code") or "").upper() in targets for w in jobs)
    return 1.0 if matches_any else 0.0


# ---------- Seniority ----------

def _years_to_level(years: int | None) -> str:
    if years is None:
        return "mid"
    if years < 3:  return "junior"
    if years < 8:  return "mid"
    if years < 15: return "senior"
    return "executive"


_LEVEL_ORDER = ("junior", "mid", "senior", "executive")


def score_seniority(bundle: dict, spec: SenioritySpec) -> float:
    if not spec.levels:
        return 0.0
    yrs = bundle["candidate"].get("years_of_experience")
    candidate_level = _years_to_level(yrs)

    current = next((w for w in bundle.get("work", []) if w.get("is_current")), None)
    if current:
        title_low = (current.get("job_title") or "").lower()
        if any(kw in title_low for kw in SENIORITY_KEYWORDS):
            idx = _LEVEL_ORDER.index(candidate_level)
            candidate_level = _LEVEL_ORDER[min(idx + 1, 3)]

    if candidate_level in spec.levels:
        return 1.0
    c_idx = _LEVEL_ORDER.index(candidate_level)
    for lvl in spec.levels:
        if abs(_LEVEL_ORDER.index(lvl) - c_idx) == 1:
            return 0.5
    return 0.0


# ---------- Skills ----------

def score_skills(bundle: dict, spec: SkillsSpec) -> float:
    if not spec.values:
        return 0.0
    target_set = {t.lower() for t in spec.values}
    candidate_skills = {(s["skill"] or "").lower(): (s.get("years_of_experience") or 0)
                        for s in bundle.get("skills", [])}
    matched = target_set & candidate_skills.keys()
    if not matched:
        return 0.0
    years_values = [candidate_skills[m] for m in matched]
    years_factor = min(sum(years_values) / (10 * len(matched)), 1.0)
    hits_fraction = len(matched) / len(target_set)
    return min(hits_fraction * (0.7 + 0.3 * years_factor), 1.0)


# ---------- Languages ----------

_PROFICIENCY_RANK = {"Beginner": 1, "Intermediate": 2, "Fluent": 3, "Native": 4}


def score_languages(bundle: dict, spec: LanguagesSpec) -> float:
    if not spec.values:
        return 0.0
    required_rank = _PROFICIENCY_RANK.get(spec.required_proficiency or "", 0)
    targets = {v.lower() for v in spec.values}
    matched = 0
    for l in bundle.get("languages", []):
        if (l["language"] or "").lower() in targets:
            if _PROFICIENCY_RANK.get(l.get("proficiency") or "", 0) >= required_rank:
                matched += 1
    return matched / len(targets) if targets else 0.0
```

- [ ] **Step 4: Run tests — expect pass**

```bash
poetry run pytest tests/test_scoring.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add app/scoring.py tests/test_scoring.py
git commit -m "feat(scoring): 6 per-dimension scoring functions with TDD tests"
```

---

### Task 14: Deterministic agent orchestration

**Files:**
- Create: `app/deterministic_agent.py`
- Test: `tests/test_deterministic_agent.py`

- [ ] **Step 1: Write orchestration tests**

Create `tests/test_deterministic_agent.py`:

```python
import datetime as dt
import pytest

from app.deterministic_agent import (
    filter_and_score,
    build_highlights,
    build_match_explanation,
)
from app.models import (
    ParsedSpec, DimensionSpec, GeoSpec, SenioritySpec, CandidateResult,
)


def _bundle(cid: str, industry: str, country: str, years: int, title: str,
            is_current: bool = True) -> dict:
    return {
        "candidate": {"id": cid, "first_name": "F", "last_name": "L",
                      "country_code": country, "nationality_code": country,
                      "years_of_experience": years, "headline": title},
        "work": [
            {"job_title": title, "start_date": dt.date(2015, 1, 1),
             "end_date": None if is_current else dt.date(2024, 1, 1),
             "is_current": is_current, "company": "X", "industry": industry,
             "company_country_code": country},
        ],
        "skills":    [],
        "languages": [],
    }


@pytest.fixture
def bundles():
    return [
        _bundle("c-match",     "Pharmaceuticals", "AE", 12, "Regulatory Affairs Director"),
        _bundle("c-wrongind",  "Finance",         "AE", 12, "Regulatory Affairs Director"),
        _bundle("c-wronggeo",  "Pharmaceuticals", "US", 12, "Regulatory Affairs Director"),
        _bundle("c-wrongrole", "Pharmaceuticals", "AE", 12, "Accountant"),
    ]


@pytest.fixture
def pharma_regulatory_me_spec():
    return ParsedSpec(
        function=DimensionSpec(values=["Regulatory Affairs"], weight=0.35, required=True),
        industry=DimensionSpec(values=["Pharmaceuticals"], weight=0.30, required=True),
        geography=GeoSpec(values=["AE", "SA", "QA"], weight=0.20, required=False,
                          location_type="current_or_nationality"),
        seniority=SenioritySpec(levels=["senior"], weight=0.15, required=False),
        temporality="any",
    )


def test_hard_filters_exclude_non_pharma(bundles, pharma_regulatory_me_spec):
    picks = filter_and_score(bundles, pharma_regulatory_me_spec, top_k=5)
    picked_ids = {p.candidate_id for p in picks}
    assert "c-wrongind" not in picked_ids          # fails industry hard filter
    assert "c-wrongrole" not in picked_ids         # fails function hard filter


def test_top_match_is_the_exact_fit(bundles, pharma_regulatory_me_spec):
    picks = filter_and_score(bundles, pharma_regulatory_me_spec, top_k=5)
    assert picks[0].candidate_id == "c-match"
    assert picks[0].per_dim is not None
    assert picks[0].per_dim["industry"] == pytest.approx(1.0, abs=0.01)


def test_empty_pool_returns_empty(pharma_regulatory_me_spec):
    picks = filter_and_score([
        _bundle("only-finance", "Finance", "US", 12, "Accountant"),
    ], pharma_regulatory_me_spec, top_k=5)
    assert picks == []


def test_highlights_include_matching_job(bundles, pharma_regulatory_me_spec):
    bundle = bundles[0]
    highlights = build_highlights(bundle, pharma_regulatory_me_spec)
    assert any("Pharmaceuticals" in h or "Regulatory" in h for h in highlights)


def test_explanation_mentions_top_dim(bundles, pharma_regulatory_me_spec):
    bundle = bundles[0]
    scores = {"function": 0.91, "industry": 1.0, "geography": 1.0, "seniority": 1.0}
    explanation = build_match_explanation(scores)
    assert "industry" in explanation.lower() or "1.0" in explanation or "1.00" in explanation
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_deterministic_agent.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/deterministic_agent.py`**

```python
"""Deterministic agent — hard filtering + weighted per-dimension scoring + highlights."""
from __future__ import annotations

from app.models import (
    ParsedSpec, CandidateResult, DimensionSpec, GeoSpec, SenioritySpec,
    SkillsSpec, LanguagesSpec,
)
from app.scoring import (
    score_function, score_industry, score_geography,
    score_seniority, score_skills, score_languages,
)


DIM_NAMES = ("function", "industry", "geography", "seniority", "skills", "languages")


def _dim(spec: ParsedSpec, name: str):
    return getattr(spec, name, None)


def _score_all(bundle: dict, spec: ParsedSpec) -> dict[str, float]:
    scores: dict[str, float] = {}
    if spec.function:  scores["function"]  = score_function(bundle, spec.function)
    if spec.industry:  scores["industry"]  = score_industry(bundle, spec.industry)
    if spec.geography: scores["geography"] = score_geography(bundle, spec.geography)
    if spec.seniority: scores["seniority"] = score_seniority(bundle, spec.seniority)
    if spec.skills:    scores["skills"]    = score_skills(bundle, spec.skills)
    if spec.languages: scores["languages"] = score_languages(bundle, spec.languages)
    return scores


def _passes_hard_filters(bundle: dict, spec: ParsedSpec, scores: dict[str, float]) -> bool:
    for name in DIM_NAMES:
        d = _dim(spec, name)
        if d is None:
            continue
        required = getattr(d, "required", False)
        if required and scores.get(name, 0.0) <= 0:
            return False
    return True


def _weights(spec: ParsedSpec) -> dict[str, float]:
    w: dict[str, float] = {}
    for name in DIM_NAMES:
        d = _dim(spec, name)
        if d is None:
            continue
        weight = getattr(d, "weight", 0.0)
        w[name] = weight
    total = sum(w.values())
    if total > 0:
        return {k: v / total for k, v in w.items()}
    return {k: 1.0 / len(w) for k in w} if w else {}


def build_match_explanation(scores: dict[str, float]) -> str:
    if not scores:
        return "No dimensions scored."
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    strong = [f"{n} ({s:.2f})" for n, s in ranked if s >= 0.8][:3]
    weak = [f"{n} ({s:.2f})" for n, s in ranked if s < 0.4][:2]
    parts = []
    if strong:
        parts.append("Strong on " + ", ".join(strong))
    if weak:
        parts.append("weak on " + ", ".join(weak))
    return "; ".join(parts) + "."


def build_highlights(bundle: dict, spec: ParsedSpec) -> list[str]:
    out: list[str] = []
    c = bundle["candidate"]

    if spec.industry and spec.industry.values:
        targets = {t.lower() for t in spec.industry.values}
        for w in bundle.get("work", []):
            if (w.get("industry") or "").lower() in targets:
                tag = "current" if w.get("is_current") else "past"
                out.append(f"{tag}: {w['job_title']} at {w['company']} ({w['industry']})")
                break

    if spec.geography and spec.geography.values:
        targets = {v.upper() for v in spec.geography.values}
        current = (c.get("country_code") or "").upper()
        if current in targets and c.get("country"):
            out.append(f"Based in {c['country']}")
        elif (c.get("nationality_code") or "").upper() in targets and c.get("nationality"):
            out.append(f"Nationality: {c['nationality']}")

    if spec.function and spec.function.values:
        targets = [t.lower() for t in spec.function.values]
        for w in bundle.get("work", []):
            title = (w.get("job_title") or "").lower()
            if any(t in title for t in targets):
                yrs = bundle["candidate"].get("years_of_experience", "?")
                out.append(f"{yrs}y as {w['job_title']}")
                break

    if spec.skills and spec.skills.values:
        skill_set = {s["skill"].lower(): s.get("years_of_experience", 0)
                     for s in bundle.get("skills", [])}
        for t in spec.skills.values:
            if t.lower() in skill_set:
                out.append(f"{t} ({skill_set[t.lower()]}y)")
                if len(out) >= 4:
                    break

    return out[:4]


def filter_and_score(bundles: list[dict], spec: ParsedSpec, top_k: int = 5) -> list[CandidateResult]:
    weights = _weights(spec)
    scored: list[tuple[float, dict, dict[str, float]]] = []
    for b in bundles:
        scores = _score_all(b, spec)
        if not _passes_hard_filters(b, spec, scores):
            continue
        final = sum(weights.get(n, 0.0) * scores.get(n, 0.0) for n in scores)
        scored.append((final, b, scores))

    scored.sort(key=lambda t: -t[0])
    results: list[CandidateResult] = []
    for rank, (final, b, scores) in enumerate(scored[:top_k], start=1):
        results.append(CandidateResult(
            candidate_id=str(b["candidate"]["id"]),
            rank=rank,
            score=round(final, 4),
            per_dim={k: round(v, 4) for k, v in scores.items()},
            match_explanation=build_match_explanation(scores),
            highlights=build_highlights(b, spec),
        ))
    return results
```

- [ ] **Step 4: Run tests — expect pass**

```bash
poetry run pytest tests/test_deterministic_agent.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/deterministic_agent.py tests/test_deterministic_agent.py
git commit -m "feat(deterministic): weighted scoring + hard filters + highlights"
```

---

## Phase 5 — Judge + Orchestrator

### Task 15: Judge

**Files:**
- Create: `app/judge.py`
- Test: `tests/test_judge.py`

- [ ] **Step 1: Write judge tests**

Create `tests/test_judge.py`:

```python
from unittest.mock import MagicMock

from app.judge import cherry_pick_top_five
from app.models import CandidateResult


def _cr(cid: str, rank: int, score: float) -> CandidateResult:
    return CandidateResult(
        candidate_id=cid, rank=rank, score=score,
        match_explanation=f"{cid} explanation",
        highlights=[f"{cid} hl1", f"{cid} hl2"],
    )


def test_judge_returns_5_results_with_reasoning():
    llm = MagicMock()
    llm.chat_json.return_value = {
        "suggested": [
            {"candidate_id": "a", "match_explanation": "both lists", "highlights": ["aa"]},
            {"candidate_id": "b", "match_explanation": "rag only",   "highlights": ["bb"]},
            {"candidate_id": "c", "match_explanation": "det only",   "highlights": ["cc"]},
            {"candidate_id": "d", "match_explanation": "both",       "highlights": ["dd"]},
            {"candidate_id": "e", "match_explanation": "semantic",   "highlights": ["ee"]},
        ],
        "reasoning": "Candidates a and d appeared in both lists...",
    }

    rag   = [_cr("a",1,0.9), _cr("b",2,0.8), _cr("f",3,0.7), _cr("e",4,0.6), _cr("g",5,0.5)]
    det   = [_cr("a",1,0.85),_cr("c",2,0.80),_cr("d",3,0.75),_cr("h",4,0.7), _cr("i",5,0.6)]
    profiles = {c: f"profile markdown for {c}" for c in "abcdefghi"}

    suggested, reasoning = cherry_pick_top_five(
        query="regulatory pharma ME",
        rag_picks=rag, det_picks=det, profile_markdown=profiles, llm=llm,
    )
    assert len(suggested) == 5
    assert suggested[0].candidate_id == "a"
    assert suggested[0].rank == 1
    assert "both" in reasoning.lower()
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_judge.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/judge.py`**

```python
"""Judge — LLM cherry-picks the final top-5 across both agents' outputs."""
from __future__ import annotations

from app.llm import LLMClient
from app.models import CandidateResult


SYSTEM_PROMPT = """You are a senior associate choosing which 5 candidates to present for a query.
Two upstream agents produced ranked candidate lists — one semantic (RAG) and one structured (SQL).

Your job: cherry-pick the best 5 candidates overall. You may pick from either list, or both.
The goal is the 5 candidates who most truly fit the query — not which agent was right.

Output MUST be valid JSON of exactly this shape:
{
  "suggested": [
    {
      "candidate_id": "<id from either input list>",
      "match_explanation": "<one concise sentence on why this candidate ranks here>",
      "highlights": ["<2-4 short proof points>", ...]
    },
    ...  (exactly 5 items)
  ],
  "reasoning": "<2-5 sentences on the overall decisions — where agents agreed, where you favored one>"
}

Output JSON only — no prose, no markdown fences."""


def _format_list(label: str, results: list[CandidateResult], profiles: dict[str, str]) -> str:
    lines = [f"=== {label} ==="]
    for r in results:
        profile = profiles.get(r.candidate_id, "(profile unavailable)")
        per_dim = ""
        if r.per_dim:
            per_dim = " | per_dim=" + ", ".join(f"{k}={v:.2f}" for k, v in r.per_dim.items())
        lines.append(
            f"[{r.rank}] candidate_id={r.candidate_id} score={r.score:.4f}{per_dim}\n"
            f"    Profile: {profile}\n"
            f"    Agent explanation: {r.match_explanation}"
        )
    return "\n".join(lines)


def cherry_pick_top_five(
    query: str,
    rag_picks: list[CandidateResult],
    det_picks: list[CandidateResult],
    profile_markdown: dict[str, str],
    llm: LLMClient,
) -> tuple[list[CandidateResult], str]:
    user = (
        f"Query:\n{query}\n\n"
        f"{_format_list('RAG (semantic)', rag_picks, profile_markdown)}\n\n"
        f"{_format_list('Deterministic (structured)', det_picks, profile_markdown)}\n\n"
        "Return your JSON now."
    )
    obj = llm.chat_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ],
        temperature=0.0,
    )

    suggested: list[CandidateResult] = []
    for rank, pick in enumerate(obj["suggested"][:5], start=1):
        suggested.append(CandidateResult(
            candidate_id=pick["candidate_id"],
            rank=rank,
            score=0.0,  # judge output doesn't carry a numeric score
            match_explanation=pick["match_explanation"],
            highlights=list(pick.get("highlights", [])),
        ))
    return suggested, obj.get("reasoning", "")
```

- [ ] **Step 4: Run tests — expect pass**

```bash
poetry run pytest tests/test_judge.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add app/judge.py tests/test_judge.py
git commit -m "feat(judge): LLM cherry-pick of final top-5 across both agents"
```

---

### Task 16: Search orchestrator

**Files:**
- Create: `app/profile_builder.py` (thin adapter around existing `profile.py`)
- Create: `app/search.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Extract profile renderer into a shared helper**

Create `app/profile_builder.py`:

```python
"""Profile markdown rendering for the judge + /experts endpoint.

Reuses the builder logic from the top-level profile.py. We duplicate the small
amount of rendering code here (instead of import-from-root) so `app/` is a clean
package without sys.path tricks. If profile.py changes, keep this in sync.
"""
from __future__ import annotations

from app.db import fetch_candidate_bundle


def _fmt_date(d) -> str:
    return d.isoformat() if d else "?"


def _full_name(c: dict) -> str:
    return f"{c['first_name']} {c['last_name']}"


def render_header(c: dict) -> str:
    parts = [f"# {_full_name(c)}"]
    if c.get("headline"):
        parts.append(f"_{c['headline']}_")
    parts.extend([
        "",
        f"- **Email:** {c.get('email','—')}",
        f"- **Phone:** {c.get('phone','—')}",
        f"- **Location:** {c.get('city','—')}, {c.get('country','—')}",
        f"- **Nationality:** {c.get('nationality','—')}",
        f"- **Years of experience:** {c.get('years_of_experience','—')}",
        f"- **ID:** `{c['id']}`",
    ])
    return "\n".join(parts)


def render_work_md(jobs: list[dict]) -> str:
    if not jobs:
        return "## Work experience\n\n_none_"
    out = [f"## Work experience ({len(jobs)})", ""]
    for j in jobs:
        end = "Present" if j.get("is_current") else _fmt_date(j.get("end_date"))
        tag = " _(current)_" if j.get("is_current") else ""
        subtitle_bits = [j.get("company", "—")]
        if j.get("industry"):        subtitle_bits.append(j["industry"])
        if j.get("company_country"): subtitle_bits.append(j["company_country"])
        out.append(f"### {j.get('job_title','—')}{tag}")
        out.append(f"_{' • '.join(subtitle_bits)}_")
        out.append(f"_{_fmt_date(j.get('start_date'))} – {end}_")
        if j.get("description"):
            out.append("")
            out.append(j["description"])
        out.append("")
    return "\n".join(out).rstrip()


def render_skills_md(skills: list[dict]) -> str:
    if not skills:
        return "## Skills\n\n_none_"
    out = [f"## Skills ({len(skills)})", ""]
    for s in skills:
        yrs = s.get("years_of_experience")
        out.append(f"- **{s['skill']}** — {yrs}y ({s.get('proficiency_level','—')})")
    return "\n".join(out)


def render_full_profile(bundle: dict) -> str:
    return "\n\n---\n\n".join([
        render_header(bundle["candidate"]),
        render_work_md(bundle["work"]),
        render_skills_md(bundle["skills"]),
    ])


def render_profile_for_id(dsn: str, candidate_id: str) -> str:
    return render_full_profile(fetch_candidate_bundle(dsn, candidate_id))


def render_mini(bundle: dict) -> str:
    """Compact single-paragraph profile for the judge prompt."""
    c = bundle["candidate"]
    current = next((w for w in bundle["work"] if w.get("is_current")), None)
    title = current["job_title"] if current else "?"
    company = current["company"] if current else "?"
    industry = current.get("industry", "—") if current else "—"
    return (
        f"{_full_name(c)} — {title} at {company} ({industry}); "
        f"{c.get('years_of_experience','?')}y exp; based in "
        f"{c.get('city','?')}, {c.get('country','?')}; nationality {c.get('nationality','?')}."
    )
```

- [ ] **Step 2: Write search orchestrator tests**

Create `tests/test_search.py`:

```python
from unittest.mock import MagicMock, patch

import numpy as np

from app.models import CandidateResult, ParsedSpec


def test_search_returns_chat_response_with_all_sections():
    from app.search import SearchService

    # Minimal parser spec
    spec = ParsedSpec(temporality="any",
                      view_weights={"summary": 0.4, "work": 0.4, "skills_edu": 0.2})

    # Mock each service called by the orchestrator
    parser_fn = MagicMock(return_value=spec)
    hyde_fn   = MagicMock(return_value="hypothetical profile text")
    embed_fn  = MagicMock(return_value=[0.1]*4)

    # Pretend 1 candidate, 3 views, dim 4
    embs = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=np.float32)
    cand_ids  = ["c1", "c1", "c1"]
    documents = ["sum", "work", "skills"]

    rag_picks = [CandidateResult(candidate_id="c1", rank=1, score=0.1,
                                 match_explanation="rag", highlights=["h1"])]
    det_picks = [CandidateResult(candidate_id="c1", rank=1, score=0.9,
                                 per_dim={"industry": 1.0},
                                 match_explanation="det", highlights=["h2"])]
    suggested = [CandidateResult(candidate_id="c1", rank=1, score=0.0,
                                 match_explanation="judge", highlights=["h3"])]

    rag_rerank = MagicMock(return_value=rag_picks)
    deterministic = MagicMock(return_value=det_picks)
    judge = MagicMock(return_value=(suggested, "because both agents agreed"))
    bundle_fn = MagicMock(return_value={
        "candidate": {"id": "c1", "first_name": "F", "last_name": "L",
                      "years_of_experience": 10, "city": "Dubai", "country": "UAE",
                      "nationality": "Emirati", "country_code": "AE",
                      "nationality_code": "AE", "headline": "h"},
        "work": [], "education": [], "skills": [], "languages": [],
    })

    svc = SearchService(
        parse_query=parser_fn,
        generate_hyde=hyde_fn,
        embed_query=embed_fn,
        all_embeddings=embs,
        candidate_ids=cand_ids,
        documents=documents,
        rag_rerank=rag_rerank,
        run_deterministic=deterministic,
        judge=judge,
        fetch_bundle=bundle_fn,
        hyde_enabled=True,
        rag_top_k=1,
    )

    resp = svc.search("find pharma experts")
    assert resp.query == "find pharma experts"
    assert resp.parsed_spec == spec
    assert len(resp.rag_picks) == 1
    assert len(resp.det_picks) == 1
    assert len(resp.suggested) == 1
    assert resp.reasoning.startswith("because")

    # Confirm HyDE was called and its output was embedded
    hyde_fn.assert_called_once()
    embed_fn.assert_called_once_with("hypothetical profile text")
```

- [ ] **Step 3: Run — expect failure**

```bash
poetry run pytest tests/test_search.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `app/search.py`**

```python
"""End-to-end orchestrator. Ties parser + RAG + deterministic + judge into one call."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.models import ChatResponse, ParsedSpec, CandidateResult
from app.rag_agent import retrieve_candidates


@dataclass
class SearchService:
    parse_query: Callable[[str], ParsedSpec]
    generate_hyde: Callable[[str], str]
    embed_query: Callable[[str], list[float]]
    all_embeddings: np.ndarray
    candidate_ids: list[str]
    documents: list[str]
    rag_rerank: Callable[[str, list[dict]], list[CandidateResult]]
    run_deterministic: Callable[[ParsedSpec], list[CandidateResult]]
    judge: Callable[[str, list[CandidateResult], list[CandidateResult], dict[str, str]],
                     tuple[list[CandidateResult], str]]
    fetch_bundle: Callable[[str], dict]
    hyde_enabled: bool = True
    rag_top_k: int = 50

    def search(self, query: str) -> ChatResponse:
        # 1. Parse query
        spec = self.parse_query(query)

        # 2. RAG: HyDE -> embed -> retrieve
        retrieval_query_text = self.generate_hyde(query) if self.hyde_enabled else query
        query_vec = np.asarray(self.embed_query(retrieval_query_text), dtype=np.float32)

        rag_pool = retrieve_candidates(
            all_embs=self.all_embeddings,
            candidate_ids=self.candidate_ids,
            documents=self.documents,
            query_vec=query_vec,
            query_text=query,
            view_weights=(dict(spec.view_weights) if spec.view_weights else None),
            top_k=self.rag_top_k,
        )
        rag_picks = self.rag_rerank(query, rag_pool)

        # 3. Deterministic
        det_picks = self.run_deterministic(spec)

        # 4. Judge — needs profiles for the union of candidate ids
        union_ids = {r.candidate_id for r in rag_picks} | {r.candidate_id for r in det_picks}
        profiles_md: dict[str, str] = {}
        for cid in union_ids:
            try:
                bundle = self.fetch_bundle(cid)
                profiles_md[cid] = _render_mini(bundle)
            except Exception:
                profiles_md[cid] = "(profile unavailable)"

        suggested, reasoning = self.judge(query, rag_picks, det_picks, profiles_md)

        return ChatResponse(
            query=query,
            parsed_spec=spec,
            rag_picks=rag_picks,
            det_picks=det_picks,
            suggested=suggested,
            reasoning=reasoning,
        )


def _render_mini(bundle: dict) -> str:
    c = bundle["candidate"]
    current = next((w for w in bundle.get("work", []) if w.get("is_current")), None)
    title = current.get("job_title", "?") if current else "?"
    company = current.get("company", "?") if current else "?"
    industry = current.get("industry", "?") if current else "?"
    return (
        f"{c.get('first_name','?')} {c.get('last_name','?')} — {title} at {company} "
        f"({industry}); {c.get('years_of_experience','?')}y exp; "
        f"based in {c.get('city','?')}, {c.get('country','?')}; "
        f"nationality {c.get('nationality','?')}."
    )
```

- [ ] **Step 5: Run tests — expect pass**

```bash
poetry run pytest tests/test_search.py -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add app/profile_builder.py app/search.py tests/test_search.py
git commit -m "feat(search): end-to-end SearchService wiring parser + RAG + det + judge"
```

---

## Phase 6 — FastAPI + CLI

### Task 17: FastAPI foundation + /health

**Files:**
- Create: `app/routes/__init__.py`
- Create: `app/routes/health.py`
- Create: `app/main.py`
- Test: `tests/test_health.py`

- [ ] **Step 1: Write /health tests**

Create `tests/test_health.py`:

```python
from fastapi.testclient import TestClient

from app.main import create_app


def test_health_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@h/d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert "checks" in body
    assert "openrouter_api_key_set" in body["checks"]
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_health.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement routes package and /health**

Create `app/routes/__init__.py`:

```python
```

Create `app/routes/health.py`:

```python
from pathlib import Path
from fastapi import APIRouter

from app.chroma_store import ChromaStore
from app.config import get_settings
from app.models import HealthResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    checks: dict[str, bool] = {
        "openrouter_api_key_set": bool(s.openrouter_api_key),
        "database_url_set":       bool(s.database_url),
        "chroma_populated":       False,
    }
    try:
        store = ChromaStore(persist_path=s.chroma_persist_path)
        checks["chroma_populated"] = store.count() > 0
    except Exception:
        pass
    status = "ok" if all(checks.values()) else "degraded"
    return HealthResponse(status=status, checks=checks)
```

- [ ] **Step 4: Implement `app/main.py`**

```python
from fastapi import FastAPI

from app.routes import health as health_routes


def create_app() -> FastAPI:
    app = FastAPI(title="Expert Search", version="0.1.0")
    app.include_router(health_routes.router)
    return app


app = create_app()
```

- [ ] **Step 5: Run tests — expect pass**

```bash
poetry run pytest tests/test_health.py -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add app/main.py app/routes/__init__.py app/routes/health.py tests/test_health.py
git commit -m "feat(api): FastAPI app scaffold with /health endpoint"
```

---

### Task 18: /ingest and /experts endpoints

**Files:**
- Create: `app/routes/ingest.py`
- Create: `app/routes/experts.py`
- Modify: `app/main.py`
- Test: `tests/test_routes_ingest.py`
- Test: `tests/test_routes_experts.py`

- [ ] **Step 1: Write /ingest tests (mock the indexer)**

Create `tests/test_routes_ingest.py`:

```python
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import create_app


def test_ingest_returns_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@h/d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))
    client = TestClient(create_app())

    with patch("app.routes.ingest.build_index", return_value={
        "candidates_indexed": 10, "documents_written": 30, "duration_seconds": 1.2,
    }):
        r = client.post("/ingest", json={"force": True})

    assert r.status_code == 200
    body = r.json()
    assert body["candidates_indexed"] == 10
    assert body["documents_written"] == 30
```

- [ ] **Step 2: Write /experts tests**

Create `tests/test_routes_experts.py`:

```python
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import create_app


def test_get_expert_returns_markdown(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@h/d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))
    client = TestClient(create_app())

    with patch("app.routes.experts.render_profile_for_id",
               return_value="# Sara Ali\n..."):
        r = client.get("/experts/abc-123")

    assert r.status_code == 200
    assert "Sara Ali" in r.json()["markdown"]
```

- [ ] **Step 3: Run — expect failure**

```bash
poetry run pytest tests/test_routes_ingest.py tests/test_routes_experts.py -v
```

Expected: ImportError on both.

- [ ] **Step 4: Implement `app/routes/ingest.py`**

```python
from fastapi import APIRouter

from app.chroma_store import ChromaStore
from app.config import get_settings
from app.ingest import build_index
from app.models import IngestRequest, IngestResponse


router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    s = get_settings()
    store = ChromaStore(persist_path=s.chroma_persist_path)
    if store.count() > 0 and not req.force:
        return IngestResponse(candidates_indexed=0, documents_written=0, duration_seconds=0.0)
    result = build_index(
        dsn=s.database_url,
        api_key=s.openrouter_api_key,
        embedding_model=s.embedding_model,
        store=store,
    )
    return IngestResponse(**result)
```

- [ ] **Step 5: Implement `app/routes/experts.py`**

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.profile_builder import render_profile_for_id


router = APIRouter()


class ExpertProfileResponse(BaseModel):
    candidate_id: str
    markdown: str


@router.get("/experts/{candidate_id}", response_model=ExpertProfileResponse)
def get_expert(candidate_id: str) -> ExpertProfileResponse:
    s = get_settings()
    try:
        md = render_profile_for_id(s.database_url, candidate_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ExpertProfileResponse(candidate_id=candidate_id, markdown=md)
```

- [ ] **Step 6: Wire both into `app/main.py`**

Update `app/main.py`:

```python
from fastapi import FastAPI

from app.routes import health as health_routes
from app.routes import ingest as ingest_routes
from app.routes import experts as experts_routes


def create_app() -> FastAPI:
    app = FastAPI(title="Expert Search", version="0.1.0")
    app.include_router(health_routes.router)
    app.include_router(ingest_routes.router)
    app.include_router(experts_routes.router)
    return app


app = create_app()
```

- [ ] **Step 7: Run tests — expect pass**

```bash
poetry run pytest tests/test_routes_ingest.py tests/test_routes_experts.py -v
```

Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add app/routes/ingest.py app/routes/experts.py app/main.py tests/test_routes_ingest.py tests/test_routes_experts.py
git commit -m "feat(api): POST /ingest and GET /experts/{id} endpoints"
```

---

### Task 19: /chat endpoint

**Files:**
- Create: `app/routes/chat.py`
- Modify: `app/main.py`
- Test: `tests/test_routes_chat.py`

- [ ] **Step 1: Write /chat test (mock the SearchService factory)**

Create `tests/test_routes_chat.py`:

```python
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import create_app
from app.models import (
    ChatResponse, ParsedSpec, CandidateResult,
)


def test_chat_returns_full_response(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@h/d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))
    client = TestClient(create_app())

    fake_svc = MagicMock()
    fake_svc.search.return_value = ChatResponse(
        query="pharma",
        parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[CandidateResult(candidate_id="c1", rank=1, score=0.1,
                                   match_explanation="rag", highlights=["x"])],
        det_picks=[CandidateResult(candidate_id="c2", rank=1, score=0.9,
                                   match_explanation="det", highlights=["y"])],
        suggested=[CandidateResult(candidate_id="c1", rank=1, score=0.0,
                                   match_explanation="j", highlights=["z"])],
        reasoning="agreement on c1",
    )
    with patch("app.routes.chat.get_search_service", return_value=fake_svc):
        r = client.post("/chat", json={"query": "pharma"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "pharma"
    assert len(body["rag_picks"]) == 1
    assert len(body["det_picks"]) == 1
    assert len(body["suggested"]) == 1
    assert body["reasoning"] == "agreement on c1"
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_routes_chat.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/routes/chat.py`**

```python
"""POST /chat — the user-facing search endpoint."""
from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter

from app.bm25_index import BM25Index
from app.chroma_store import ChromaStore
from app.config import get_settings
from app.db import fetch_candidate_bundle, fetch_all_bundles
from app.deterministic_agent import filter_and_score
from app.embeddings import EmbeddingClient
from app.hyde import generate_hypothetical_profile
from app.judge import cherry_pick_top_five
from app.llm import LLMClient
from app.models import ChatRequest, ChatResponse
from app.query_parser import parse_query
from app.rag_agent import rerank_and_explain
from app.search import SearchService


router = APIRouter()


@lru_cache(maxsize=1)
def _load_index():
    """Load Chroma + bundles once per process. Cached."""
    s = get_settings()
    store = ChromaStore(persist_path=s.chroma_persist_path)
    embs, metas, docs, ids = store.load_all()
    # Re-order into (candidate, view_order) if needed. Assume ingestion wrote in that order.
    candidate_ids = [m["candidate_id"] for m in metas]
    bundles = fetch_all_bundles(s.database_url)
    bundles_by_id = {str(b["candidate"]["id"]): b for b in bundles}
    return embs, candidate_ids, docs, bundles_by_id


def get_search_service() -> SearchService:
    s = get_settings()
    llm = LLMClient(api_key=s.openrouter_api_key, model=s.llm_model)
    embedder = EmbeddingClient(api_key=s.openrouter_api_key, model=s.embedding_model)

    embs, candidate_ids, documents, bundles_by_id = _load_index()

    def _fetch_bundle(cid: str) -> dict:
        if cid in bundles_by_id:
            return bundles_by_id[cid]
        return fetch_candidate_bundle(s.database_url, cid)

    return SearchService(
        parse_query=lambda q: parse_query(q, llm=llm),
        generate_hyde=lambda q: generate_hypothetical_profile(q, llm=llm),
        embed_query=lambda t: embedder.embed_one(t),
        all_embeddings=embs,
        candidate_ids=candidate_ids,
        documents=documents,
        rag_rerank=lambda q, pool: rerank_and_explain(q, pool, llm=llm),
        run_deterministic=lambda spec: filter_and_score(
            [b for b in bundles_by_id.values()], spec,
            top_k=s.deterministic_top_k,
        ),
        judge=lambda q, rp, dp, pm: cherry_pick_top_five(q, rp, dp, pm, llm=llm),
        fetch_bundle=_fetch_bundle,
        hyde_enabled=s.hyde_enabled,
        rag_top_k=s.rag_top_k,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    service = get_search_service()
    return service.search(req.query)
```

- [ ] **Step 4: Wire into `app/main.py`**

Update `app/main.py`:

```python
from fastapi import FastAPI

from app.routes import health as health_routes
from app.routes import ingest as ingest_routes
from app.routes import experts as experts_routes
from app.routes import chat as chat_routes


def create_app() -> FastAPI:
    app = FastAPI(title="Expert Search", version="0.1.0")
    app.include_router(health_routes.router)
    app.include_router(ingest_routes.router)
    app.include_router(experts_routes.router)
    app.include_router(chat_routes.router)
    return app


app = create_app()
```

- [ ] **Step 5: Run tests — expect pass**

```bash
poetry run pytest tests/test_routes_chat.py -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add app/routes/chat.py app/main.py tests/test_routes_chat.py
git commit -m "feat(api): POST /chat — full search pipeline behind FastAPI"
```

---

### Task 20: CLI companion

**Files:**
- Create: `app/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write CLI tests (mock the services)**

Create `tests/test_cli.py`:

```python
import json
from unittest.mock import MagicMock, patch

from app.cli import main
from app.models import ChatResponse, ParsedSpec, CandidateResult


def test_cli_chat_prints_json(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@h/d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))

    svc = MagicMock()
    svc.search.return_value = ChatResponse(
        query="q",
        parsed_spec=ParsedSpec(temporality="any"),
        rag_picks=[], det_picks=[], suggested=[],
        reasoning="none",
    )
    with patch("app.cli.get_search_service", return_value=svc):
        exit_code = main(["chat", "pharma experts"])
    assert exit_code == 0

    out = capsys.readouterr().out
    # CLI prints JSON lines
    parsed = json.loads(out.splitlines()[-1])
    assert parsed["query"] == "q"


def test_cli_ingest_prints_summary(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:y@h/d")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", str(tmp_path / "chroma"))

    with patch("app.cli.build_index", return_value={
        "candidates_indexed": 42, "documents_written": 126, "duration_seconds": 5.0,
    }):
        exit_code = main(["ingest", "--force"])
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "42" in out and "126" in out
```

- [ ] **Step 2: Run — expect failure**

```bash
poetry run pytest tests/test_cli.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `app/cli.py`**

```python
"""CLI companion. Same core services as the API, easier for local testing."""
from __future__ import annotations

import argparse
import json
import sys

from app.chroma_store import ChromaStore
from app.config import get_settings
from app.ingest import build_index
from app.routes.chat import get_search_service


def cmd_chat(args) -> int:
    service = get_search_service()
    resp = service.search(args.query)
    print(json.dumps(resp.model_dump(), indent=2, default=str))
    return 0


def cmd_ingest(args) -> int:
    s = get_settings()
    store = ChromaStore(persist_path=s.chroma_persist_path)
    if store.count() > 0 and not args.force:
        print("Index already populated; use --force to rebuild.", file=sys.stderr)
        return 1
    result = build_index(
        dsn=s.database_url,
        api_key=s.openrouter_api_key,
        embedding_model=s.embedding_model,
        store=store,
    )
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="expert-search")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_chat = sub.add_parser("chat")
    p_chat.add_argument("query")
    p_chat.set_defaults(func=cmd_chat)

    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("--force", action="store_true")
    p_ing.set_defaults(func=cmd_ingest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests — expect pass**

```bash
poetry run pytest tests/test_cli.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/cli.py tests/test_cli.py
git commit -m "feat(cli): chat + ingest subcommands reusing same core services"
```

---

## Phase 7 — Integration + README

### Task 21: End-to-end integration test

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_end_to_end.py`

- [ ] **Step 1: Write a runnable integration test (uses real DB + real LLM)**

Create `tests/integration/__init__.py`:

```python
```

Create `tests/integration/test_end_to_end.py`:

```python
"""Smoke test — hits real Postgres + real OpenRouter. Skip locally with `-m 'not integration'`.

Assumes: index has been built (run `poetry run python -m app.cli ingest --force`).
"""
import os
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


pytestmark = pytest.mark.integration


def test_chat_end_to_end(monkeypatch):
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    client = TestClient(create_app())
    r = client.post("/chat", json={
        "query": "regulatory affairs experts in pharma in the Middle East",
    })
    assert r.status_code == 200
    body = r.json()

    # Shape checks
    assert body["query"].startswith("regulatory")
    assert "parsed_spec" in body
    assert isinstance(body["rag_picks"], list)
    assert isinstance(body["det_picks"], list)
    assert isinstance(body["suggested"], list)
    assert isinstance(body["reasoning"], str)

    # Each result carries the brief-required fields
    for r_ in body["suggested"]:
        assert "candidate_id" in r_
        assert "match_explanation" in r_
        assert "highlights" in r_
```

- [ ] **Step 2: Run (only if env is set up)**

```bash
poetry run pytest tests/integration/test_end_to_end.py -v -m integration
```

Expected: 1 passed (or skipped if `OPENROUTER_API_KEY` not set).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/
git commit -m "test(integration): end-to-end /chat smoke test on real services"
```

---

### Task 22: README + model-choice rationale

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

Create `README.md`:

```markdown
# Expert Search — Applied AI Take-Home

FastAPI service that accepts natural-language queries and returns top-5 candidates from a Postgres candidate DB,
ranked by a hybrid (RAG + deterministic SQL) retrieval system with a judge-synthesized best-of-both.

## Setup

```bash
# 1. Install deps
poetry install

# 2. Configure environment
cp .env.example .env
# Edit .env — fill in DATABASE_URL (password) and OPENROUTER_API_KEY

# 3. Build the vector index (one-off, ~3-5 minutes for 10K candidates)
poetry run python -m app.cli ingest --force

# 4. Run the API
poetry run uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | Build/rebuild the Chroma index |
| `POST` | `/chat` | Submit a NL query; return ranked experts |
| `GET`  | `/health` | Liveness + readiness checks |
| `GET`  | `/experts/{candidate_id}` | Full candidate profile as Markdown |

## Example

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "regulatory affairs experts in pharma in the Middle East"}' | jq
```

Response (abridged):

```json
{
  "query": "regulatory affairs experts in pharma in the Middle East",
  "parsed_spec": { "industry": { "values": ["Pharmaceuticals"], "required": true, ... }, ... },
  "rag_picks":  [{ "candidate_id": "...", "rank": 1, "score": 0.034,
                   "match_explanation": "...", "highlights": ["..."] }, ...],
  "det_picks":  [{ "candidate_id": "...", "rank": 1, "score": 0.82,
                   "per_dim": { "industry": 1.0, "geography": 1.0, ... },
                   "match_explanation": "...", "highlights": ["..."] }, ...],
  "suggested":  [{ "candidate_id": "...", "rank": 1,
                   "match_explanation": "...", "highlights": ["..."] }, ...],
  "reasoning": "Both agents agreed on..."
}
```

## Design

See [`docs/superpowers/specs/2026-04-20-expert-search-design.md`](docs/superpowers/specs/2026-04-20-expert-search-design.md).

**High level:**

```
Query → LLM Parser → { RAG agent (semantic), Deterministic agent (SQL) } → Judge → top 5 with reasoning
```

**Key choices:**

- **Chroma** (SQLite-backed) for the vector store — no Docker, simple local persistence.
- **rank_bm25** for BM25 — trivial tokenization, transparent scoring.
- **Full-scan hybrid retrieval** (not top-K) — at 30K vectors, one numpy matmul is ~20ms and gives complete per-candidate info.
- **RRF fusion** over vector + BM25 — normalization-free, robust.
- **Listwise LLM rerank** (not pointwise) — one LLM call instead of 50; relative judgments are empirically better.
- **Deterministic agent with weighted dim scoring** — `function`, `industry`, `geography`, `seniority`, `skills`, `languages`. Deliberate signal weighting, per the brief.

## Model choice rationale

- **Chat/reasoning model** (`LLM_MODEL`, default `anthropic/claude-3.5-sonnet`): used for query parsing, HyDE, listwise rerank, and judge. Chosen for strong reasoning + JSON reliability. Swap to Haiku for cost-sensitive deployment.
- **Embedding model** (`EMBEDDING_MODEL`, default `openai/text-embedding-3-small`): 1536-dim, cheap (~$0.02/1M tokens), widely used. Any OpenRouter-compatible embedding works.

## Testing

```bash
# Unit tests (fast, no external services)
poetry run pytest -v -m "not integration"

# Integration tests (real DB + real LLM)
poetry run pytest -v -m integration
```

## Out of scope for this iteration

- **Conversational context** (follow-up queries like *"Filter those to Saudi only"*). The `ChatRequest` schema accepts `conversation_id` but it's currently a no-op.
- **Part 3 — Evaluation & Precision Thinking** — a written deliverable from the brief, intentionally deferred.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, endpoints, design summary, model-choice rationale"
```

---

## Self-review

1. **Spec coverage:** every brief-required endpoint (`/chat`, `/ingest`, `/health`) has a task (Task 17, 18, 19); every scoring dim (function, industry, geography, seniority, skills, languages) has a test + impl (Task 13); RAG (Tasks 11-12) and deterministic (Task 14) agents both produce `match_explanation` + `highlights` per the updated spec §5.2/§5.3; judge cherry-picks (Task 15); env-based config with `.env.example` (Task 1); Pydantic v2 models (Task 2); README + model rationale (Task 22). Deferred items (conversation_id, Part 3) are explicitly out-of-scope per the spec.

2. **Placeholder scan:** all TDD steps contain complete code. One explicit `# TODO: user writes 5-10 if-then rules here` exists in the query-parser system prompt (Task 9 step 3) — this is the user-supplied domain input the spec calls out as intentional, not a plan omission.

3. **Type consistency:** `CandidateResult` carries `per_dim` as `dict[str, float] | None`; deterministic agent populates it, RAG leaves it None — consistent across Task 2, 12, 14. `ParsedSpec.view_weights` is `dict[Literal["summary","work","skills_edu"], float] | None` in Task 2, consumed as such in Task 11 and Task 16. `ChatResponse` fields (`query, parsed_spec, rag_picks, det_picks, suggested, reasoning`) used identically across Task 2, Task 16, Task 19.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-20-expert-search.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
