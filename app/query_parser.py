"""NL query → ParsedSpec via a single structured-output LLM call."""
from __future__ import annotations

from app.llm import LLMClient
from app.models import ParsedSpec


SYSTEM_PROMPT = """You are a candidate-search query parser. Convert the user's
natural-language query into a structured ParsedSpec.

Schema conventions (enforced by strict JSON Schema — you must stay within them):
- For each dimension (function, industry, geography, seniority, skills, languages),
  set it to `null` when the query doesn't mention it, or to its object form when it does.
- `temporality` is one of "current", "past", "any".
- `geography.values` uses ISO 3166-1 alpha-2 country codes (two uppercase letters).
- `geography.location_type` is one of "current", "current_or_nationality", "historical".
- `seniority.levels` items are "junior" | "mid" | "senior" | "executive".
- `view_weights` (optional) is a dict with keys "summary", "work", "skills_edu" summing to ~1.0.

Entity-expansion rules:
- "Middle East" -> ["AE", "SA", "QA", "BH", "KW", "OM", "EG", "JO", "LB"]
- "GCC" -> ["AE", "SA", "QA", "BH", "KW", "OM"]
- "MENA" -> ["AE", "SA", "QA", "BH", "KW", "OM", "EG", "JO", "LB", "MA", "DZ", "TN", "LY"]
- "pharma" / "pharmaceutical" -> ["Pharmaceuticals", "Biotechnology"]
- "petrochemical" -> ["Oil & Gas", "Chemicals", "Petroleum Products"]
- Silently normalize typos (e.g. "pharmacuetical" -> "Pharmaceutical").

Required-vs-nice-to-have:
- Only set `required: true` when the query uses hard language: "must", "only",
  "specifically", a specific company name, "exactly". Default is `required: false`.

Weights:
- All dimension weights should sum to <= 1.0.
- Weight reflects emphasis in the query: if the query is mostly about industry
  (e.g. "pharma experts"), industry.weight is higher. Generic filler phrasing
  gets a lower weight on everything.

view_weights (drives RAG view aggregation):
- Work-history-heavy queries ("former CPO", "worked at", "led a team at")
  -> `{summary: 0.25, work: 0.55, skills_edu: 0.20}`
- Skill-heavy ("Python expert", "knows Kubernetes")
  -> `{summary: 0.30, work: 0.20, skills_edu: 0.50}`
- Balanced / generic ("find me X")
  -> `{summary: 0.40, work: 0.35, skills_edu: 0.25}`
- When unsure, leave view_weights null.

Temporality:
- "former [role]" / "ex-[role]" / "used to be" -> "past"
- "currently [role]" / "now [role]" -> "current"
- Otherwise -> "any"

CUSTOM WEIGHTING HEURISTICS (user-supplied — fill in with domain rules):
# TODO: user writes 5–10 if-then rules here. Examples:
# - If query mentions "junior" / "entry-level" / "new grad":
#   seniority.levels=["junior"], geography.required=false, geography.weight<=0.1
# - If query specifies a company name (e.g. "worked at Pfizer"):
#   industry is inferred from the company; treat the company as a hard filter
#   by including it in function.values and setting required=true.
# - If query mentions "must have N+ years":
#   min_years_exp=N, and add a hidden implicit senior weight.
"""


def parse_query(query: str, llm: LLMClient) -> ParsedSpec:
    """Single structured LLM call. Returns a validated ParsedSpec."""
    return llm.chat_structured(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ],
        response_model=ParsedSpec,
        temperature=0.0,
        max_tokens=800,
    )
