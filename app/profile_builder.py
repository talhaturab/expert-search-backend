"""Profile markdown rendering for the judge + /experts endpoint.

The exploration/profile.py script contains an older, richer version of this. We
duplicate a minimal renderer here (instead of importing from the top-level
`exploration/` tree) so `app/` is a clean package with no sys.path tricks.
"""
from __future__ import annotations

from app.db import fetch_candidate_bundle


def _fmt_date(d) -> str:
    return d.isoformat() if d else "?"


def _full_name(c: dict) -> str:
    return f"{c.get('first_name','?')} {c.get('last_name','?')}"


# ---------- Mini profile (used by the judge prompt) ----------

def render_mini(bundle: dict) -> str:
    """Compact single-paragraph profile — cheap to include 10 of these in a
    judge prompt without blowing the context."""
    c = bundle["candidate"]
    current = next((w for w in bundle.get("work", []) if w.get("is_current")), None)
    if current:
        title = current.get("job_title", "?")
        company = current.get("company", "?")
        industry = current.get("industry") or "—"
        current_str = f"{title} at {company} ({industry})"
    else:
        current_str = "no current role"
    yrs = c.get("years_of_experience", "?")
    loc = f"{c.get('city','?')}, {c.get('country','?')}"
    nat = c.get("nationality", "?")
    headline = c.get("headline") or ""
    return (
        f"{_full_name(c)} — {current_str}; {yrs}y exp; "
        f"based in {loc}; nationality {nat}."
        + (f" Headline: {headline}" if headline else "")
    )


# ---------- Full profile (used by GET /experts/{id}) ----------

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


def render_education_md(edu: list[dict]) -> str:
    if not edu:
        return "## Education\n\n_none_"
    out = [f"## Education ({len(edu)})", ""]
    for e in edu:
        line = f"- {e.get('degree','?')} in {e.get('field','?')} — {e.get('institution','?')}"
        if e.get("start_year") or e.get("graduation_year"):
            line += f" ({e.get('start_year','?')} – {e.get('graduation_year','?')})"
        if e.get("grade"):
            line += f", grade {e['grade']}"
        out.append(line)
    return "\n".join(out)


def render_languages_md(langs: list[dict]) -> str:
    if not langs:
        return "## Languages\n\n_none_"
    out = [f"## Languages ({len(langs)})", ""]
    for l in langs:
        out.append(f"- **{l['language']}** — {l.get('proficiency','—')}")
    return "\n".join(out)


def render_full_profile(bundle: dict) -> str:
    return "\n\n---\n\n".join([
        render_header(bundle["candidate"]),
        render_work_md(bundle.get("work", [])),
        render_skills_md(bundle.get("skills", [])),
        render_education_md(bundle.get("education", [])),
        render_languages_md(bundle.get("languages", [])),
    ])


def render_profile_for_id(dsn: str, candidate_id: str) -> str:
    return render_full_profile(fetch_candidate_bundle(dsn, candidate_id))
