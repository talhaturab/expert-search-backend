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
