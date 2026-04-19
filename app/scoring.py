"""Per-dimension scoring functions for the deterministic agent. Each returns [0, 1]."""
from __future__ import annotations

import datetime as dt
from difflib import SequenceMatcher

from app.models import (
    DimensionSpec, GeoSpec, SenioritySpec, SkillsSpec, LanguagesSpec,
)


SENIORITY_KEYWORDS = ("chief", "vp", "vice president", "head", "director", "principal", "lead")
_LEVEL_ORDER = ("junior", "mid", "senior", "executive")
_PROFICIENCY_RANK = {"Beginner": 1, "Intermediate": 2, "Fluent": 3, "Native": 4}


# ---------- helpers ----------

def _trgm_sim(a: str, b: str) -> float:
    """Cheap trigram-like similarity — stand-in for pg_trgm on Python side."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _years_of(job: dict) -> float:
    start = job.get("start_date")
    end = job.get("end_date") or dt.date.today()
    if not start:
        return 0.0
    return max(0.0, (end - start).days / 365.25)


# ---------- Function ----------

def score_function(bundle: dict, spec: DimensionSpec) -> float:
    """Best fuzzy title/headline match × recency weight."""
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

def score_industry(bundle: dict, spec: DimensionSpec) -> float:
    """Fraction of career years spent in matching industries."""
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
    """Country match per `location_type`. Binary — 0 or 1."""
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
    # historical — any past/current job's company in a target country
    jobs = bundle.get("work", [])
    if any((w.get("company_country_code") or "").upper() in targets for w in jobs):
        return 1.0
    return 0.0


# ---------- Seniority ----------

def _years_to_level(years: int | None) -> str:
    if years is None:
        return "mid"
    if years < 3:  return "junior"
    if years < 8:  return "mid"
    if years < 15: return "senior"
    return "executive"


def score_seniority(bundle: dict, spec: SenioritySpec) -> float:
    """Map years → level, upgrade by title keyword, 1.0 if in target, 0.5 if adjacent."""
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
    """Hits fraction × (0.7 + 0.3 × years factor)."""
    if not spec.values:
        return 0.0
    target_set = {t.lower() for t in spec.values}
    candidate_skills = {
        (s["skill"] or "").lower(): (s.get("years_of_experience") or 0)
        for s in bundle.get("skills", [])
    }
    matched = target_set & candidate_skills.keys()
    if not matched:
        return 0.0
    years_values = [candidate_skills[m] for m in matched]
    years_factor = min(sum(years_values) / (10 * len(matched)), 1.0)
    hits_fraction = len(matched) / len(target_set)
    return min(hits_fraction * (0.7 + 0.3 * years_factor), 1.0)


# ---------- Languages ----------

def score_languages(bundle: dict, spec: LanguagesSpec) -> float:
    """Fraction of target languages held at or above required proficiency."""
    if not spec.values:
        return 0.0
    required_rank = _PROFICIENCY_RANK.get(spec.required_proficiency or "", 0)
    targets = {v.lower() for v in spec.values}
    matched = 0
    for l in bundle.get("languages", []):
        if (l.get("language") or "").lower() in targets:
            if _PROFICIENCY_RANK.get(l.get("proficiency") or "", 0) >= required_rank:
                matched += 1
    return matched / len(targets) if targets else 0.0
