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


def _word_subset(a: str, b: str) -> bool:
    """True if every word in `a` appears as a whole word in `b` (case-insensitive).

    Catches length-asymmetric matches that trigram similarity misses:
    ("Security", "Security Assurance") → True.  ("AI", "AIDS") → False because
    "AIDS" is a single word, not a word containing "AI".

    Also catches seniority-prefix cases: ("Data Scientist", "Senior Data
    Scientist") → True, which trigram alone scores ~0.83 — the same value as
    genuinely-different pairs like ("QA Engineer", "Data Engineer").
    """
    a_words = a.lower().split()
    b_words = set(b.lower().split())
    return bool(a_words) and all(w in b_words for w in a_words)


def _years_of(job: dict) -> float:
    start = job.get("start_date")
    end = job.get("end_date") or dt.date.today()
    if not start:
        return 0.0
    return max(0.0, (end - start).days / 365.25)


# ---------- Function ----------

# Role-suffix words that appear at the end of many different job titles and
# therefore don't identify a function on their own. "QA Engineer" and "Data
# Engineer" both contain "engineer" — that overlap shouldn't count as a match.
_GENERIC_TITLE_WORDS = frozenset({
    "engineer", "manager", "analyst", "developer", "specialist",
    "administrator", "coordinator", "consultant", "director",
    "officer", "executive", "associate", "representative", "agent",
    "assistant", "scientist",
})


def _function_similarity(target: str, title: str) -> float:
    """Similarity score in [0, 1] between a target function and a title.

    Three tiers:
    1. Word-subset match (target ⊂ title or vice versa) → 1.0. Handles the
       common case of a generic target ("Frontend Developer") vs a seniority-
       qualified title ("Senior Frontend Developer").
    2. If the target has content words (non-generic suffixes) but none overlap
       with the candidate, treat as a false-positive trigram match and cap at
       0.5. Prevents "Data Engineer" scoring 0.83 against "QA Engineer" just
       because they share the generic word "Engineer".
    3. Otherwise return trigram similarity (preserves typo tolerance).
    """
    if not target or not title:
        return 0.0
    if _word_subset(target, title) or _word_subset(title, target):
        return 1.0

    t_content = set(target.lower().split()) - _GENERIC_TITLE_WORDS
    c_content = set(title.lower().split()) - _GENERIC_TITLE_WORDS
    if t_content and not (t_content & c_content):
        # High character overlap but no shared non-generic word — cap hard.
        return min(_trgm_sim(target, title), 0.5)

    return _trgm_sim(target, title)


def score_function(bundle: dict, spec: DimensionSpec) -> float:
    """Best title/headline match × recency weight."""
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
            sim = _function_similarity(target, title) * rec
            if sim > best:
                best = sim
    return min(best, 1.0)


# ---------- Industry ----------

# Someone currently in the target industry is a stronger "find-me-someone-at-X"
# signal than tenure fraction alone — a recent switcher currently at a pharma
# company should outrank a 20-year pharma veteran who just moved to fintech.
# 0.7 is high enough to lift currently-in-target candidates into contention
# without overwhelming a 100% tenure match.
_CURRENT_ROLE_INDUSTRY_FLOOR = 0.7


def score_industry(bundle: dict, spec: DimensionSpec) -> float:
    """Industry match = max(current-role floor, tenure fraction).

    - Currently in a target industry → score ≥ 0.7 regardless of tenure.
    - Otherwise → fraction of career years in target industries.
    """
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
    tenure = min(matched / total, 1.0) if total > 0 else 0.0

    current = next((w for w in bundle.get("work", []) if w.get("is_current")), None)
    if current and (current.get("industry") or "").lower() in targets:
        return max(_CURRENT_ROLE_INDUSTRY_FLOOR, tenure)
    return tenure


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

# Skills is a big-vocab field (1,551 distinct values) — we don't inject the list
# into the parser prompt. Instead we match each LLM-emitted target skill against
# the candidate's actual skills with exact + word-subset + trigram fallback.
_SKILL_FUZZY_THRESHOLD = 0.80


def _best_fuzzy_match(target: str, candidate_skills: dict[str, int]) -> tuple[str, int] | None:
    """Return (candidate_skill_name, years) of the best match, or None.

    Precedence: word-subset containment > trigram similarity. Word-subset catches
    short-target / long-candidate cases like "Security" ⊂ "Security Assurance"
    that trigram (ratio 0.62) rejects against our 0.80 threshold.
    """
    for name in candidate_skills:
        if _word_subset(target, name) or _word_subset(name, target):
            return name, candidate_skills[name]

    best, best_score = None, 0.0
    for name in candidate_skills:
        s = _trgm_sim(target, name)
        if s > best_score:
            best, best_score = name, s
    if best is not None and best_score >= _SKILL_FUZZY_THRESHOLD:
        return best, candidate_skills[best]
    return None


def score_skills(bundle: dict, spec: SkillsSpec) -> float:
    """Hits fraction × (0.7 + 0.3 × years factor).

    Each target skill is matched against the candidate's actual skills:
    - exact case-insensitive match preferred,
    - trigram fuzzy match (≥ 0.80 similarity) as fallback.
    """
    if not spec.values:
        return 0.0
    candidate_skills = {
        (s["skill"] or "").lower(): (s.get("years_of_experience") or 0)
        for s in bundle.get("skills", [])
    }
    if not candidate_skills:
        return 0.0

    matched_years: list[int] = []
    for target in spec.values:
        target_low = target.lower()
        if target_low in candidate_skills:
            matched_years.append(candidate_skills[target_low])
            continue
        m = _best_fuzzy_match(target_low, candidate_skills)
        if m is not None:
            matched_years.append(m[1])

    if not matched_years:
        return 0.0
    years_factor = min(sum(matched_years) / (10 * len(matched_years)), 1.0)
    hits_fraction = len(matched_years) / len(spec.values)
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
