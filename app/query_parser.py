"""NL query → ParsedSpec via a single structured-output LLM call.

Optionally ground the parser against the DB's real vocabulary so the LLM picks
exact strings (industries, skill categories, languages, proficiency) that match
downstream SQL lookups without any fuzzy post-matching.
"""
from __future__ import annotations

from app.llm import LLMClient
from app.models import ParsedSpec, PriorContext
from app.vocabulary import Vocabulary


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


def _filter_to_known(values: list[str], known: list[str]) -> list[str]:
    """Keep only values that appear (case-insensitively) in `known`, using the
    known casing in the output. Unknown values are dropped."""
    known_by_lower = {k.lower(): k for k in known}
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        canonical = known_by_lower.get(v.lower())
        if canonical and canonical not in seen:
            out.append(canonical)
            seen.add(canonical)
    return out


def _restrict_to_vocabulary(spec: ParsedSpec, vocab: Vocabulary) -> None:
    """Post-validate the LLM's output: drop any value not in the DB vocabulary.

    Only applies to fields whose vocab we injected into the prompt. Skills and
    function titles use fuzzy matching downstream and are left as-is here.
    """
    if spec.industry is not None:
        spec.industry.values = _filter_to_known(spec.industry.values, vocab.industries)
    if spec.languages is not None:
        spec.languages.values = _filter_to_known(spec.languages.values, vocab.languages)
        if spec.languages.required_proficiency is not None:
            # Proficiency is a Literal enum in the Pydantic schema already; validate
            # against vocab as a second line of defence.
            if spec.languages.required_proficiency not in vocab.proficiency_levels:
                spec.languages.required_proficiency = None  # type: ignore[assignment]


def _render_prior_context_block(prior: PriorContext) -> str:
    return (
        "\n\nPRIOR TURN CONTEXT:\n"
        f"- prior query: {prior.prior_query!r}\n"
        f"- prior result candidate IDs: {prior.prior_suggested_ids}\n"
        f"- prior parsed spec: {prior.prior_parsed_spec.model_dump(exclude_none=True)}\n"
        "\nREFINEMENT DETECTION:\n"
        "Set `is_refinement: true` ONLY if the new query narrows or filters the\n"
        "prior result set (phrases like 'filter those to ...', 'among them, only ...',\n"
        "'narrow to ...', 'from those, only ...'). A brand-new search that happens to\n"
        "share a topic is NOT a refinement.\n"
        "\n"
        "If `is_refinement: true`, produce a ParsedSpec containing ONLY the NEW\n"
        "constraints the user is adding. Do NOT re-emit the prior spec's constraints —\n"
        "the search will restrict to the prior candidate pool."
    )


def parse_query(
    query: str,
    llm: LLMClient,
    vocab: Vocabulary | None = None,
    prior_context: PriorContext | None = None,
) -> ParsedSpec:
    """Single structured LLM call. Returns a validated (and optionally
    vocab-grounded) ParsedSpec.
    """
    system = SYSTEM_PROMPT
    if vocab is not None:
        system = system + "\n\n" + vocab.to_prompt_block()
    if prior_context is not None:
        system = system + _render_prior_context_block(prior_context)

    spec = llm.chat_structured(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ],
        response_model=ParsedSpec,
        temperature=0.0,
    )

    if vocab is not None:
        _restrict_to_vocabulary(spec, vocab)
    return spec
