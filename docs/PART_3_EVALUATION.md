# Part 3 — Evaluation & Precision Thinking

*Written deliverable for the InfoQuest Applied AI take-home brief, Part 3.*

This document addresses the brief's three sub-questions: how to build a ground-truth eval dataset, what metrics to track, and a walk-through of one concrete failure mode — a real bug I observed while manually grading the current system's outputs against 5 test queries (see `evaluation_testing/MANUAL_JUDGING_REPORT.md` for the raw evaluation).

---

## 1. Ground Truth Design

Assume we have 100+ historical engagements where consulting clients received an expert shortlist and went on to book, interview, or reject each candidate. Each engagement is a natural **query → delivered shortlist → labelled outcomes** triple — richer than synthetic ground truth because the labels reflect real commercial decisions, not armchair judgement.

### Fields I'd capture per engagement

| Field | Why it matters |
|---|---|
| **`engagement_id`** | Primary key; links to the CRM/engagement system |
| **`query_text`** | The original client brief as stated, verbatim |
| **`query_parsed`** | The system's `ParsedSpec` output, stored so we can later measure parser drift |
| **`query_metadata`** | Client name, industry, geo, urgency, time of year — needed for stratified sampling and drift detection |
| **`delivered_shortlist`** | The list of `candidate_id`s actually sent to the client, in presentation order |
| **`per_candidate_outcome`** | One of: `booked`, `interviewed`, `reviewed_rejected`, `not_reviewed`, `client_specifically_negative`. This is the label |
| **`client_satisfaction`** | Post-engagement: 1–5 rating, binary "would use shortlist again" |
| **`notes`** | Free-text from ops explaining why a particular candidate was great/bad — grist for failure-analysis mining |

### Defining "correct"

Treating this as a single-label classification problem ("was each candidate right or wrong?") breaks for two reasons: (a) clients rarely articulate their full preference up-front, and (b) `not_reviewed` isn't the same as `rejected`. Three-tier labelling avoids both traps:

1. **Gold** — candidates who were `booked` or `interviewed`. These are ground-truth positives: the system got it right *and* the real world confirmed it.
2. **Acceptable** — candidates in `delivered_shortlist` with outcome `reviewed_rejected`. Ops thought they were worth showing; the client had taste reasons to decline. Counting these as positives would reward mediocre retrieval; counting them as negatives punishes the system for ops's own judgement. We treat them as **neutral** in metrics.
3. **Negative** — candidates explicitly flagged `client_specifically_negative` ("this person is our competitor's CTO — never show them again"). Harder to collect but incredibly valuable for precision-killer tests.

**Important caveat on what this ground truth proves.** A historical engagement tells us *these specific candidates were good* — it doesn't tell us that *no other candidate would have been good*. Our dataset is therefore a **lower bound** on the set of correct answers, not an exhaustive list. Practically: if the system returns a candidate who is not in the historical shortlist, we can't conclude they're wrong — we only know they weren't tested. This shapes the metrics: we should over-weight *recall against known Gold* (if the system fails to surface a known-booked candidate for the same query, that's definitely a regression) and interpret "miss rate" on new candidates cautiously (could be novelty, could be error).

### Sampling + splits

- Stratify by query shape (hard-criteria, open-ended, fuzzy) so the eval set doesn't over-represent one kind of query.
- Hold out 20% per stratum as a locked test set; the rest is dev. Never tune against the test split.
- Re-draw the splits every 3–6 months to handle vocabulary and talent-pool drift — last year's "senior ML engineer" is today's "Staff AI Engineer."

### Scaling from 100 to 1,000+ engagements

Active learning: the system flags its own **low-confidence** outputs (parser unsure of `required` flags, deterministic and RAG disagreeing sharply, judge reasoning mentions "none are ideal fits") for ops to label first. These are the examples that move metrics the most per labelling-hour spent.

---

## 2. Precision Metrics

### Which metrics

| Metric | What it captures | Why it's here |
|---|---|---|
| **Precision@5** | Of the 5 recommendations, how many are Gold? | Our output is always top-5 and this is the metric our UX is optimised for |
| **MRR@5** (Mean Reciprocal Rank) | Position of the first Gold candidate | Rewards putting the best answer at #1, which the brief explicitly cares about (*"We care less about how many results come back and more about whether the right ones are at the top"*) |
| **NDCG@5** | Graded relevance, rank-discounted | Handles the three-tier Gold/Acceptable/Negative labelling cleanly; Acceptable contributes less than Gold, Negative subtracts |
| **False-positive rate** (bad picks per query) | How often a `client_specifically_negative` shows up in top-5 | The one metric where a single bad case is a fireable offence — tracked separately from aggregate averages |

I would report **P@5 as the primary headline** because it maps 1:1 to what a senior associate actually experiences, and **MRR@5 as the secondary** because the top of the list carries disproportionate decision weight.

### Why precision over recall

The senior-associate workflow dictates this.

- **Their time is the binding constraint.** They have five minutes to read five profiles and make a call. If four of the five are wrong, the senior associate either rejects the entire shortlist (expensive) or trusts a weak pick (worse). Recall — "did we find every qualified candidate in the 10K pool?" — is invisible to them.
- **The cost function is asymmetric.** A bad recommendation makes the consulting firm look like they don't understand the client's business. One such recommendation reaches the client's CEO. A missed good candidate, by contrast, is recoverable: the associate runs a second query, adjusts a filter, or asks the ops team.
- **False positives poison trust; false negatives don't.** Once a senior associate has seen one truly off-topic candidate (our "Museum Administrator with a life-sciences degree" in Q5 of the evaluation), they stop reading carefully. Precision is how we preserve the social capital that lets this tool be used at all.
- **Recall is unbounded; precision is tractable.** We never know the denominator of "every good candidate in the 10K pool" — multiple could fit. We *do* know the numerator for precision: "how many of the 5 we returned were right."

Concretely in our system, the three design choices that sacrifice recall for precision are: (a) the deterministic agent's **hard filters on `required: true` dimensions** (can return empty rather than degrade); (b) the listwise rerank's **honest scoring** where `score < 40` candidates still surface with their low score rather than being hidden; (c) the judge's **opt-in to empty outputs** when neither agent's picks clear the bar (seen in practice — see failure analysis below).

### Reporting cadence and rollback thresholds

Metrics alone don't buy trust — a cadence does.

- **Per-PR**: run P@5 + MRR@5 on the locked test set before merging any change to parser, scoring, or rerank. **Block merge on a regression of > 3 percentage points** on either metric without an explicit override + written rationale. Cheap enough at 100+ engagements to run on every PR.
- **Weekly**: re-run on the full labelled set, stratified by query type. Watch for a single stratum regressing (e.g. precision collapsing on "former X at Y company" queries while overall P@5 holds) — aggregates hide these.
- **On every production /chat**: log the `parsed_spec`, `per_dim` scores, and judge reasoning. These are free telemetry for post-hoc failure analysis — the Q2 bug described in §3 below was found by a human reading such logs, not by a metric.
- **Rollback rule**: any week-over-week drop of > 5 pp on P@5 for any single stratum pulls the new release. Trust compounds; it's cheap to pause, expensive to rebuild.

---

## 3. Failure Analysis — a real case from our evaluation

**Query:** *"Former CPO at a Saudi petrochemical company"* — this is a real query from our manual-judging report; using an observed failure (rather than invented one) makes the analysis load-bearing rather than hypothetical.

**The profile that got ranked highly when it shouldn't have.** The deterministic agent's **#1 pick was Daniel Ibrahim** — currently a Senior PM at Roche/Onyx/Carlyle in Washington DC, Lebanese nationality, with a past `Lead Operations Manager` stint at the Saudi Ministry of Municipal and Rural Affairs and Housing. He got a deterministic score of 0.74 with `per_dim = {industry: 0.0, geography: 1.0, seniority: 1.0, skills: 1.0}`. Flagged #1 confidently; visually ranks higher than Wei Zhang (a currently-at-Saudi-Aramco candidate whom the RAG agent surfaced).

He is not wrong *on paper*: Saudi ministry experience, senior-level, recent product management. But he's wrong *for this query* — the client asked for **petrochemical**, not government, and Daniel has zero oil/gas/chemicals exposure anywhere in his career. The other four deterministic picks had the same signature: strong on geography + seniority, `per_dim.industry = 0.0`.

**The aggregate finding.** Every single Deterministic-agent pick had `per_dim.industry = 0.0`, despite the DB containing candidates with genuine petrochemical / oil & gas history (Wei Zhang at Saudi Aramco; Sami Watanabe's past VP role). The deterministic agent correctly identified Saudi-geography candidates and C-level titles, but it silently scored zero on the most load-bearing dimension — industry — and nobody noticed until a human read the results.

**Root cause.** A vocabulary mismatch between the parser's emitted industry string and the DB's actual `companies.industry` values.

- The parser's system prompt contains an expansion rule: `"petrochemical" -> ["Oil & Gas", "Chemicals", "Petroleum Products"]` (ampersand, title case).
- The DB's `companies.industry` column actually uses `"Oil and Gas"` (the word "and", not an ampersand).
- `score_industry` in `app/scoring.py` does a case-insensitive set-membership check (`(w.get("industry") or "").lower() in targets`). `"oil and gas"` is not in `{"oil & gas", "chemicals", "petroleum products"}` → zero.
- Because the dimension was `required=false` (the parser softly weighted it), the candidate still passed the hard filter gate and still got a final score > 0 from *other* dimensions (geography + seniority). So the pick "succeeded" with a broken foundation.
- `per_dim={'industry': 0.0, ...}` was exposed in the API response — but nothing in the system looked at it. It's a silent signal that only a human judge caught.

**Why it's interesting.** This is not an LLM mistake or a retrieval bug — both the parser and the scorer did *exactly* what the code said. The bug lives in the gap between two well-specified components.

**The fix.** Two layers:

1. **Tighten vocabulary grounding at the parser.** We already inject the DB's distinct industries into the system prompt (see `app/vocabulary.py`). Today `_restrict_to_vocabulary` only *filters* unknown values — if the LLM emits `"Oil & Gas"` and the DB uses `"Oil and Gas"`, the bad value is dropped, leaving `industry.values=[]`, which scores 0 the same way. The fix is to **add a post-filter fuzzy canonicalisation pass**: for each dropped value, find the nearest vocab entry by trigram similarity with a 0.85 threshold, substitute it. `"Oil & Gas"` → `"Oil and Gas"` would canonicalise cleanly; `"Blockchain Miner"` wouldn't (no neighbour above threshold).

2. **Raise a warning, don't silently score zero.** When `score_industry` gets a non-empty `target_set` but zero overlap with any candidate's work history AND zero overlap is suspicious *for that query* (e.g., industry had `weight > 0.20`), emit a structured warning in the `ChatResponse` metadata: `{"warnings": [{"dim": "industry", "reason": "all candidates scored 0 — check parser→DB vocabulary mismatch"}]}`. A senior associate who sees the warning knows to check the `per_dim` breakdowns before trusting the shortlist.

**What this teaches about our evaluation design.** The fix to this bug couldn't come from P@5 alone — the numerical metric wouldn't reveal *why* precision was low for this query. Ground truth labelling at the per-dimension level (engagements should annotate "which facet of the query was hardest to satisfy?") is what converts a precision regression into a diagnosable failure. That's the argument for capturing `query_parsed` in the ground-truth schema: it lets us regression-test the parser end-to-end against real client briefs, not just candidates.
