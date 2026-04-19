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
