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


# ---------------------------------------------------------------------------
# Bulk versions of the per-candidate queries — fetch EVERYTHING in one round-trip
# then group by candidate_id in Python. Avoids the N+1 problem.
# ---------------------------------------------------------------------------

_BULK_WORK_SQL = WORK_SQL.replace(
    "WHERE we.candidate_id = %s\nORDER BY we.is_current DESC, we.start_date DESC",
    "ORDER BY we.candidate_id, we.is_current DESC, we.start_date DESC",
)
_BULK_EDUCATION_SQL = EDUCATION_SQL.replace(
    "WHERE e.candidate_id = %s\nORDER BY e.graduation_year DESC NULLS LAST",
    "ORDER BY e.candidate_id, e.graduation_year DESC NULLS LAST",
)
_BULK_SKILLS_SQL = SKILLS_SQL.replace(
    "WHERE cs.candidate_id = %s\nORDER BY cs.years_of_experience DESC NULLS LAST, s.name",
    "ORDER BY cs.candidate_id, cs.years_of_experience DESC NULLS LAST, s.name",
)
_BULK_LANGUAGES_SQL = LANGUAGES_SQL.replace(
    "WHERE cl.candidate_id = %s\nORDER BY pl.rank DESC, l.name",
    "ORDER BY cl.candidate_id, pl.rank DESC, l.name",
)

# The bulk queries still need candidate_id in the SELECT so we can group afterward.
for _before, _after in [
    ("SELECT we.job_title,",       "SELECT we.candidate_id, we.job_title,"),
    ("SELECT e.start_year,",       "SELECT e.candidate_id, e.start_year,"),
    ("SELECT s.name AS skill,",    "SELECT cs.candidate_id, s.name AS skill,"),
    ("SELECT l.name AS language,", "SELECT cl.candidate_id, l.name AS language,"),
]:
    _BULK_WORK_SQL      = _BULK_WORK_SQL.replace(_before, _after, 1)
    _BULK_EDUCATION_SQL = _BULK_EDUCATION_SQL.replace(_before, _after, 1)
    _BULK_SKILLS_SQL    = _BULK_SKILLS_SQL.replace(_before, _after, 1)
    _BULK_LANGUAGES_SQL = _BULK_LANGUAGES_SQL.replace(_before, _after, 1)


def _bulk(cur, sql: str) -> dict[str, list[dict]]:
    """Run a bulk query and group rows by candidate_id (popping the key from each row)."""
    cur.execute(sql)
    out: dict[str, list[dict]] = {}
    for row in cur.fetchall():
        row = dict(row)
        cid = str(row.pop("candidate_id"))
        out.setdefault(cid, []).append(row)
    return out


def fetch_all_bundles(dsn: str, limit: int | None = None) -> list[dict]:
    """All (or first N) candidates with full bundles, in 5 round-trips total.

    1 query for the enriched candidate list, then 4 bulk queries for work /
    education / skills / languages. Rows grouped by candidate_id in Python.
    Replaces the old 4×N N+1 loop (which stalled at 40K queries).
    """
    candidates = fetch_all_candidates(dsn)
    if limit is not None:
        candidates = candidates[:limit]

    with psycopg2.connect(dsn) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            work_by_cid      = _bulk(cur, _BULK_WORK_SQL)
            education_by_cid = _bulk(cur, _BULK_EDUCATION_SQL)
            skills_by_cid    = _bulk(cur, _BULK_SKILLS_SQL)
            languages_by_cid = _bulk(cur, _BULK_LANGUAGES_SQL)

    bundles: list[dict] = []
    for c in candidates:
        cid = str(c["id"])
        bundles.append({
            "candidate": c,
            "work":      work_by_cid.get(cid, []),
            "education": education_by_cid.get(cid, []),
            "skills":    skills_by_cid.get(cid, []),
            "languages": languages_by_cid.get(cid, []),
        })
    return bundles
