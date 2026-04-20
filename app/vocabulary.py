"""Vocabulary loader — fetches distinct DB values to ground the LLM parser.

Small enums (industries / skill_categories / languages / proficiency_levels) are
injected into the parser prompt so the LLM picks exact DB strings. Big vocab
(skills — 1,551 values) is NOT injected but cached for downstream fuzzy matching.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import psycopg2


@dataclass
class Vocabulary:
    """Distinct values from the DB, loaded once and shared across requests."""
    industries: list[str] = field(default_factory=list)
    skill_categories: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    proficiency_levels: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)  # not injected into prompt; for fuzzy match

    def to_prompt_block(self) -> str:
        """Compact ground-truth vocab block to append to the parser's system prompt."""
        return (
            "Known DB vocabulary — when the query implies one of these, "
            "you MUST pick the EXACT string from the list (no new values, no paraphrases):\n\n"
            f"INDUSTRIES: {', '.join(self.industries)}\n\n"
            f"SKILL CATEGORIES: {', '.join(self.skill_categories)}\n\n"
            f"LANGUAGES: {', '.join(self.languages)}\n\n"
            f"PROFICIENCY LEVELS: {', '.join(self.proficiency_levels)}\n"
        )


def _fetch_distinct(cur, sql: str) -> list[str]:
    cur.execute(sql)
    return sorted({row[0] for row in cur.fetchall() if row[0] is not None})


def load_vocabulary(dsn: str) -> Vocabulary:
    """One DB round-trip per vocab field. Call at app startup / after /ingest."""
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            industries = _fetch_distinct(cur,
                "SELECT DISTINCT industry FROM companies WHERE industry IS NOT NULL")
            skill_categories = _fetch_distinct(cur, "SELECT name FROM skill_categories")
            languages = _fetch_distinct(cur, "SELECT name FROM languages")
            proficiency_levels = _fetch_distinct(cur,
                "SELECT name FROM proficiency_levels ORDER BY rank")
            skills = _fetch_distinct(cur, "SELECT name FROM skills")
    return Vocabulary(
        industries=industries,
        skill_categories=skill_categories,
        languages=languages,
        proficiency_levels=proficiency_levels,
        skills=skills,
    )
