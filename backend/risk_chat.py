"""Risk Chat — the decision-support agent for the Risk page.

``answer(question, applicant_id=None, history=None) -> dict`` mirrors
``concierge.answer`` exactly:

  1. ROUTER   — ``route`` deterministically classifies the question into one of
                explain | whatif | counterfactual | compare | exclusions |
                general.
  2. TOOLS    — ``risk_agent`` runs the pure tools over ``risk.py`` and returns
                EXACT structured numbers (probability, band, reason codes,
                governance facts). The agent never computes these itself.
  3. SYNTHESIZE — Claude (via ``llm.get_langchain_llm``) rewrites ONLY the
                assembled numbered context into prose with inline [n] citations
                (source="anthropic"). Offline / on any error we emit a templated
                grounded answer (source="rules").

DECISION-SUPPORT ONLY: the answers never approve, deny, price, or condition a
lease — an elevated estimate "routes to a person". Like ``concierge.answer``
this NEVER raises: the outermost guard always returns a safe response dict.

All intents are fully implemented: explain / exclusions / general (Phase 1) and
what-if / counterfactual / compare (Phase 2). Each Phase-2 intent runs its own
``risk_agent`` tool over a modified COPY of the profile (never persisted) and
builds the matching discriminated-union artifact; what-if and counterfactual
answers are always flagged EXPLORATORY (not the saved estimate). A tool hiccup
degrades to the grounded explain view rather than failing.
"""

from __future__ import annotations

import re

from llm import get_langchain_llm
import risk
import risk_agent


# ---------------------------------------------------------------------------
# ROUTER — intent classification
# ---------------------------------------------------------------------------
# Governance / feature-policy intent: "what do you use", "what's excluded",
# "is X used", fairness / bias / protected-class questions. Checked FIRST so a
# question naming a feature is treated as governance, not scored explanation.
_EXCLUSIONS_RE = re.compile(
    r"\b("
    r"exclud|not used|don'?t use|do(?:es)? (?:it|the model|you) use|is .* used|"
    r"which features|what features|what factors|feature[s]? (?:used|considered)|"
    r"allowed features|inputs?|variables?|"
    r"fair|fairness|bias|biased|discriminat|protected|proxy|redlin|"
    r"model card|governance|compliant|complian|legal|lawful|"
    r"race|ethnic|national origin|gender|\bsex\b|disab|religion|"
    r"familial|marital|dependents?|children|\bage\b|\bstudent\b|criminal|"
    r"\bpets?\b|smoker|location|neighborhood"
    r")\b",
    re.IGNORECASE,
)

# Counterfactual intent: "what would it take to lower / reach a band".
_COUNTERFACTUAL_RE = re.compile(
    r"("
    r"what would it take|what needs to change|what (?:can|could|should) .* (?:do|change) to|"
    r"how (?:can|could|do|would) .* (?:lower|reduce|improve|decrease|bring down)|"
    r"(?:get|move|drop|bring) .* (?:to|into|out of|below) .* (?:low|medium|high|band)|"
    r"reach .* band|counterfactual|what would (?:lower|reduce|improve)"
    r")",
    re.IGNORECASE,
)

# What-if intent: hypothetical single-lever changes.
_WHATIF_RE = re.compile(
    r"("
    r"what if|what happens if|suppose|hypothetical|"
    r"if (?:the|their|they|his|her|income|credit|rent|dti|savings) .* "
    r"(?:was|were|had|is|increase|decrease|change|higher|lower|goes)|"
    r"would the (?:score|estimate|probability) change"
    r")",
    re.IGNORECASE,
)

# Compare intent: this applicant vs the portfolio / others.
_COMPARE_RE = re.compile(
    r"("
    r"\bcompare\b|\bversus\b|\bvs\.?\b|percentile|\brank\b|ranking|"
    r"portfolio|other applicant|others|the average|vs\.? (?:the )?average|"
    r"how does .* compare|relative to|compared to|typical applicant"
    r")",
    re.IGNORECASE,
)

# Explain intent: why is this score what it is / drivers / band.
_EXPLAIN_RE = re.compile(
    r"\b("
    r"why|explain|reason|driver|factor|elevated|high[\s-]?risk|risk score|"
    r"probability|\bband\b|how risky|what'?s driving|what is driving|"
    r"break ?down|understand|so high|so low"
    r")\b",
    re.IGNORECASE,
)


def route(question: str, applicant_id: str | None = None) -> str:
    """Classify the question into an intent.

    Precedence (per the contract): governance/exclusions first, then
    counterfactual, what-if, compare, and finally explain — which is the
    default when an applicant is scoped, else ``general``.
    """
    q = question or ""
    if _EXCLUSIONS_RE.search(q):
        return "exclusions"
    if _COUNTERFACTUAL_RE.search(q):
        return "counterfactual"
    if _WHATIF_RE.search(q):
        return "whatif"
    if _COMPARE_RE.search(q):
        return "compare"
    if _EXPLAIN_RE.search(q):
        return "explain"
    return "explain" if applicant_id else "general"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _pct(v) -> str:
    try:
        return f"{round(float(v) * 100)}%"
    except (TypeError, ValueError):
        return "n/a"


def _snippet(text: str, n: int = 220) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _range_str(result: dict) -> str:
    rng = result.get("range") or []
    if len(rng) == 2:
        return f"{_pct(rng[0])}–{_pct(rng[1])}"
    return "n/a"


def _factors(result: dict) -> tuple[list[str], list[str]]:
    """(increase-labels, decrease-labels) from the model's own reason codes."""
    ups = [r.get("label", "") for r in result.get("reason_codes", []) if r.get("direction") == "increases"]
    downs = [r.get("label", "") for r in result.get("reason_codes", []) if r.get("direction") == "decreases"]
    return [l for l in ups if l], [l for l in downs if l]


def _num(v) -> str:
    """A tidy numeric string for a raw feature value (int when whole)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f == int(f) else f"{f:.2f}"


def _signed_pts(delta) -> str:
    """A signed percentage-point delta, e.g. '+4 points' / '-3 points'."""
    try:
        pts = round(float(delta) * 100)
    except (TypeError, ValueError):
        return "n/a"
    return f"{'+' if pts > 0 else ''}{pts} points"


# ---------------------------------------------------------------------------
# CONTEXT ASSEMBLY + SOURCES (parallel lists — sources[i] backs citation [i+1])
# ---------------------------------------------------------------------------
def _explain_context(result: dict) -> tuple[list[str], list[dict]]:
    """A risk-result block + a key-factors block for the explain intent."""
    name = result.get("name") or "This applicant"
    conf = result.get("confidence", "high")
    result_line = (
        f"Risk estimate — {name}: estimated probability of late payment "
        f"{_pct(result.get('probability'))}, {result.get('band')} band, "
        f"{conf} confidence, plausible range {_range_str(result)}. "
        f"Source: {result.get('source')} ({result.get('model_type')})."
    )
    ups, downs = _factors(result)
    factor_line = (
        "Key factors (the model's own reason codes). "
        + (f"Increase the estimate: {'; '.join(ups)}. " if ups else "Increase the estimate: none notable. ")
        + (f"Lower the estimate: {'; '.join(downs)}." if downs else "Lower the estimate: none notable.")
    )
    context_blocks = [
        f"Risk estimate — {name}:\n{result_line}",
        f"Key factors — {name}:\n{factor_line}",
    ]
    sources = [
        {
            "type": "risk",
            "label": f"Risk estimate — {name}",
            "snippet": _snippet(result_line),
            "applicant_id": result.get("applicant_id", ""),
        },
        {
            "type": "factors",
            "label": f"Key factors — {name}",
            "snippet": _snippet(factor_line),
            "applicant_id": result.get("applicant_id", ""),
        },
    ]
    return context_blocks, sources


def _governance_context(gov: dict) -> tuple[list[str], list[dict]]:
    """A model-card/governance block (+ a lookup block when the question named
    specific fields) for the exclusions intent."""
    card = gov.get("card") or {}
    used = card.get("features") or []
    excluded = card.get("excluded") or []
    card_line = (
        f"Model card — {card.get('name', 'Resident Late-Payment Risk')}. "
        f"Intended use: {card.get('intended_use', '')} "
        f"The model uses {len(used)} legitimate financial / payment factors and "
        f"structurally excludes {len(excluded)} protected-class proxy or "
        f"non-predictive fields. It never uses race, national origin, sex, "
        f"familial status, disability, age, or location."
    )
    context_blocks = [f"Model card / feature policy:\n{card_line}"]
    sources = [
        {
            "type": "model_card",
            "label": f"Model card — {card.get('name', 'Resident Late-Payment Risk')}",
            "snippet": _snippet(card_line),
            "applicant_id": "",
        }
    ]

    lookup = gov.get("feature_lookup") or []
    if lookup:
        parts = []
        for item in lookup:
            verb = "IS used" if item.get("status") == "used" else "is NOT used (excluded)"
            parts.append(f"'{item.get('field')}' {verb} — {item.get('reason')}")
        lookup_line = "Feature policy lookup. " + " ".join(parts)
        context_blocks.append(f"Feature policy lookup:\n{lookup_line}")
        sources.append(
            {
                "type": "model_card",
                "label": "Feature policy lookup",
                "snippet": _snippet(lookup_line),
                "applicant_id": "",
            }
        )
    return context_blocks, sources


def _general_context() -> tuple[list[str], list[dict]]:
    """A short capability/overview block for the portfolio-scope general intent
    (keeps any LLM prose grounded; the deterministic answer stands alone)."""
    card = risk.model_card()
    line = (
        "Risk decision-support assistant. It can explain an applicant's "
        "estimated late-payment probability and the model's own reason codes, "
        "and describe the feature policy (what is used and what is excluded, and "
        f"why). {card.get('intended_use', '')}"
    )
    context_blocks = [f"About the risk assistant:\n{line}"]
    sources = [
        {
            "type": "model_card",
            "label": "About the risk assistant",
            "snippet": _snippet(line),
            "applicant_id": "",
        }
    ]
    return context_blocks, sources


def _whatif_context(wf: dict) -> tuple[list[str], list[dict]]:
    """A hypothetical-estimate block + a feature-changes block for the what-if
    intent. Both are explicitly flagged exploratory / not the saved estimate."""
    base = wf["base"]
    whatif = wf["whatif"]
    name = whatif.get("name") or "This applicant"
    aid = whatif.get("applicant_id", "") or base.get("applicant_id", "")
    applied = wf.get("overrides") or {}
    applied_txt = ", ".join(f"{k} = {_num(v)}" for k, v in applied.items()) or "none"
    delta = wf.get("delta_probability", 0.0)
    direction = "lower" if delta < 0 else ("higher" if delta > 0 else "no change in")
    band_note = (
        f"the band changes from {base['band']} to {whatif['band']}"
        if wf.get("band_changed")
        else f"it stays in the {whatif['band']} band"
    )
    result_line = (
        f"Hypothetical what-if — {name} (EXPLORATORY; not the saved estimate). "
        f"Applied change(s): {applied_txt}. Saved estimate {_pct(base['probability'])} "
        f"({base['band']} band); hypothetical estimate {_pct(whatif['probability'])} "
        f"({whatif['band']} band) — {_signed_pts(delta)} ({direction} risk); {band_note}."
    )
    changes = wf.get("features_changed") or []
    change_txt = (
        "; ".join(f"{c['label']} (from {_num(c['from'])} to {_num(c['to'])})" for c in changes)
        if changes
        else "no model features moved"
    )
    factor_line = f"What-if feature changes (model inputs only): {change_txt}."
    context_blocks = [
        f"What-if estimate — {name}:\n{result_line}",
        f"What-if feature changes — {name}:\n{factor_line}",
    ]
    sources = [
        {"type": "risk", "label": f"What-if estimate — {name}", "snippet": _snippet(result_line), "applicant_id": aid},
        {"type": "factors", "label": f"What-if feature changes — {name}", "snippet": _snippet(factor_line), "applicant_id": aid},
    ]
    return context_blocks, sources


def _counterfactual_context(cf: dict) -> tuple[list[str], list[dict]]:
    """A target/feasibility block + an ordered-levers block for the
    counterfactual intent. Flagged exploratory / illustrative, not advice."""
    base = cf["base"]
    final = cf["final"]
    name = base.get("name") or "This applicant"
    aid = base.get("applicant_id", "")
    tb = cf.get("target_band", "medium")
    steps = cf.get("steps") or []
    reach = "Reaches" if cf.get("achievable") else "Does not reach"
    result_line = (
        f"Counterfactual — {name} (EXPLORATORY illustration; not the saved estimate and not advice). "
        f"Saved estimate {_pct(base['probability'])} ({base['band']} band); target {tb} band. "
        f"{reach} the {tb} band using the model's financial levers within realistic bounds "
        f"({cf.get('predict_calls', 0)} scenarios screened). Projected estimate after changes: "
        f"{_pct(final['probability'])} ({final['band']} band)."
    )
    if steps:
        steps_txt = "; ".join(
            f"{s['label']} → {_pct(s['probability_after'])} ({s['band_after']} band)" for s in steps
        )
    else:
        steps_txt = "no single modeled change lowers the estimate within realistic bounds"
    lever_line = f"Modeled changes, in order of impact: {steps_txt}."
    context_blocks = [
        f"Counterfactual target — {name}:\n{result_line}",
        f"Counterfactual levers — {name}:\n{lever_line}",
    ]
    sources = [
        {"type": "risk", "label": f"Counterfactual — {name}", "snippet": _snippet(result_line), "applicant_id": aid},
        {"type": "factors", "label": f"Counterfactual levers — {name}", "snippet": _snippet(lever_line), "applicant_id": aid},
    ]
    return context_blocks, sources


def _compare_context(cmp: dict) -> tuple[list[str], list[dict]]:
    """A position-vs-portfolio block + a distribution block for the compare
    intent (applicant scope)."""
    result = cmp["result"]
    ps = cmp["portfolio"]
    name = result.get("name") or "This applicant"
    aid = result.get("applicant_id", "")
    bc = ps.get("band_counts") or {}
    scored = ps.get("scored", 0)
    result_line = (
        f"Portfolio comparison — {name}. Estimated probability {_pct(result['probability'])} "
        f"({result['band']} band). Portfolio average is {_pct(ps.get('avg_probability'))} across "
        f"{scored} scored applicants; this applicant is {_signed_pts(cmp.get('vs_avg'))} vs the "
        f"average and sits at about the {cmp.get('percentile')}th percentile (rank "
        f"{cmp.get('band_rank')} of {scored} by estimated risk, where rank 1 is highest)."
    )
    dist_line = (
        f"Portfolio band distribution: {bc.get('low', 0)} low, {bc.get('medium', 0)} medium, "
        f"{bc.get('high', 0)} high; high-risk share {ps.get('high_risk_pct')}%. Median estimate "
        f"{_pct(ps.get('median_probability'))}."
    )
    context_blocks = [
        f"Portfolio comparison — {name}:\n{result_line}",
        f"Portfolio distribution:\n{dist_line}",
    ]
    sources = [
        {"type": "risk", "label": f"Portfolio comparison — {name}", "snippet": _snippet(result_line), "applicant_id": aid},
        {"type": "risk", "label": "Portfolio distribution", "snippet": _snippet(dist_line), "applicant_id": ""},
    ]
    return context_blocks, sources


def _portfolio_context(ps: dict) -> tuple[list[str], list[dict]]:
    """A portfolio-wide summary block for the compare intent at portfolio scope
    (no applicant selected)."""
    bc = ps.get("band_counts") or {}
    line = (
        f"Portfolio risk summary across {ps.get('scored', 0)} scored applicants: average "
        f"estimate {_pct(ps.get('avg_probability'))}, median {_pct(ps.get('median_probability'))}, "
        f"high-risk share {ps.get('high_risk_pct')}%. Bands: {bc.get('low', 0)} low, "
        f"{bc.get('medium', 0)} medium, {bc.get('high', 0)} high. Select an applicant to compare "
        f"one against this distribution."
    )
    context_blocks = [f"Portfolio risk summary:\n{line}"]
    sources = [
        {"type": "risk", "label": "Portfolio risk summary", "snippet": _snippet(line), "applicant_id": ""}
    ]
    return context_blocks, sources


def _band_of(p) -> str:
    """Band for a probability (reuses the risk_agent threshold helper)."""
    try:
        return risk_agent._band_of(float(p))
    except (TypeError, ValueError):
        return "low"


def _portfolio_rows(ps: dict) -> list[dict]:
    """Reference RiskComparisonRow points for the comparison artifact: the
    portfolio average and median."""
    avg = ps.get("avg_probability", 0.0)
    median = ps.get("median_probability", 0.0)
    return [
        {"label": "Portfolio average", "probability": avg, "band": _band_of(avg)},
        {"label": "Portfolio median", "probability": median, "band": _band_of(median)},
    ]


# ---------------------------------------------------------------------------
# FOLLOW-UP SUGGESTIONS (deterministic — work with the LLM offline)
# ---------------------------------------------------------------------------
_FOLLOWUPS_EXPLAIN = [
    "What factors does the model use?",
    "What's excluded from the model and why?",
    "How are the risk bands defined?",
]
_FOLLOWUPS_EXCLUSIONS = [
    "Why is credit score used?",
    "Is criminal history used?",
    "How are the risk bands defined?",
]
_FOLLOWUPS_GENERAL = [
    "What factors does the model use?",
    "What's excluded from the model and why?",
    "How does the risk model work?",
]
_FOLLOWUPS_WHATIF = [
    "What would move them to a lower band?",
    "How does this compare to the portfolio?",
    "Why is the estimate at this level?",
]
_FOLLOWUPS_COUNTERFACTUAL = [
    "What if their income were higher?",
    "How does this compare to the portfolio?",
    "What factors does the model use?",
]
_FOLLOWUPS_COMPARE = [
    "Why is this applicant's estimate at this level?",
    "What would move them to a lower band?",
    "What factors does the model use?",
]


def _follow_ups(intent: str) -> list[str]:
    if intent == "exclusions":
        return list(_FOLLOWUPS_EXCLUSIONS)
    if intent == "general":
        return list(_FOLLOWUPS_GENERAL)
    if intent == "whatif":
        return list(_FOLLOWUPS_WHATIF)
    if intent == "counterfactual":
        return list(_FOLLOWUPS_COUNTERFACTUAL)
    if intent == "compare":
        return list(_FOLLOWUPS_COMPARE)
    return list(_FOLLOWUPS_EXPLAIN)


# ---------------------------------------------------------------------------
# SYSTEM PROMPTS + guardrails
# ---------------------------------------------------------------------------
_RISK_SYSTEM = (
    "You are the Resident Late-Payment Risk assistant — DECISION-SUPPORT ONLY. "
    "Answer using ONLY the numbered context provided; never invent or recompute "
    "a probability, band, range, or reason code. Cite the context you use inline "
    "with bracketed numbers like [1] or [2]. Explain risk through the model's OWN "
    "reason codes. You MUST NEVER approve, deny, reject, accept, price, or "
    "condition a lease, and never say whether someone should be rented to — an "
    "elevated estimate simply routes the application to a person for review. "
    "Refuse to use or speculate about excluded or protected attributes and cite "
    "the model card's reason. Note low confidence when present (missing inputs "
    "are neutral-imputed and never raise risk). The model is trained on synthetic "
    "data; results are an estimate, not a guarantee, and not a consumer report. "
    "Be concise and factual."
)

_GOVERNANCE_SYSTEM = (
    "You are the Resident Late-Payment Risk assistant explaining the model's "
    "FEATURE POLICY / governance. Answer using ONLY the numbered context; never "
    "invent features, reasons, or metrics. Cite inline with [n]. State plainly "
    "which fields are used (legitimate financial / payment factors) and which are "
    "structurally excluded (protected-class proxies or non-predictive fields), "
    "and give the model card's reason. Never approve, deny, price, or condition a "
    "lease. This is decision-support on synthetic data — an estimate, not a "
    "consumer report. Be concise and factual."
)


_WHATIF_SYSTEM = (
    _RISK_SYSTEM
    + " This turn is a HYPOTHETICAL what-if. Make clear it is EXPLORATORY and does "
    "NOT change the applicant's saved estimate. Report only the numbers in the "
    "context — the saved estimate, the hypothetical estimate, the point change, "
    "and which model inputs moved — and never suggest the change should be made or "
    "what decision follows from it."
)

_COUNTERFACTUAL_SYSTEM = (
    _RISK_SYSTEM
    + " This turn is a counterfactual: an EXPLORATORY, illustrative list of changes "
    "to the model's financial inputs that the model projects would lower the "
    "estimate toward a target band. It is NOT advice, a promise, or a change to the "
    "saved estimate. Describe only the modeled levers and projected estimates in the "
    "context; never tell anyone to take these actions or imply an outcome is "
    "guaranteed, and never approve, deny, or condition anything."
)

_COMPARE_SYSTEM = (
    _RISK_SYSTEM
    + " This turn compares one applicant against the scored portfolio. Report only "
    "the relative position in the context — percentile, distance from the average, "
    "rank, and band distribution — as neutral context for a human reviewer, never as "
    "a ranking that approves, denies, or prioritizes anyone."
)


def _system_for(intent: str) -> str:
    if intent == "exclusions":
        return _GOVERNANCE_SYSTEM
    if intent == "whatif":
        return _WHATIF_SYSTEM
    if intent == "counterfactual":
        return _COUNTERFACTUAL_SYSTEM
    if intent == "compare":
        return _COMPARE_SYSTEM
    return _RISK_SYSTEM


# ---------------------------------------------------------------------------
# DETERMINISTIC ANSWERS (per-intent templates; NO approve/deny lexicon)
# ---------------------------------------------------------------------------
_DEFLECTION = (
    "I don't have a saved risk estimate for that applicant. Select an applicant "
    "from the list to see their estimated late-payment probability and the "
    "factors behind it, or ask me about the model's feature policy — what it "
    "uses and what it excludes."
)


def _deterministic_explain(result: dict, note: str = "") -> str:
    name = result.get("name") or "This applicant"
    ups, downs = _factors(result)
    lead = (
        f"{name}'s estimated probability of paying rent late is "
        f"{_pct(result.get('probability'))} — the {result.get('band')} band [1] "
        f"(plausible range {_range_str(result)}, {result.get('confidence')} confidence)."
    )
    factors = ""
    if ups or downs:
        up_txt = f"raised mainly by {', '.join(ups)}" if ups else ""
        down_txt = f"offset by {', '.join(downs)}" if downs else ""
        joiner = ", " if up_txt and down_txt else ""
        factors = f" It is {up_txt}{joiner}{down_txt} [2]."
    conf_note = ""
    if result.get("confidence") == "low":
        conf_note = (
            " Confidence is low because some inputs were missing and neutral-"
            "imputed — that never raises the estimate."
        )
    guard = (
        " This is decision-support only: an elevated estimate routes the "
        "application to a person for review, not to any automated decision."
    )
    tail = f" {note}" if note else ""
    return lead + factors + conf_note + guard + tail


def _deterministic_exclusions(gov: dict) -> str:
    lookup = gov.get("feature_lookup") or []
    parts: list[str] = []
    if lookup:
        for item in lookup:
            if item.get("status") == "used":
                parts.append(
                    f"'{item.get('field')}' IS used — {item.get('reason')} [1]"
                )
            else:
                parts.append(
                    f"'{item.get('field')}' is NOT used — {item.get('reason')} [1]"
                )
    card = gov.get("card") or {}
    used = card.get("features") or []
    base = (
        f"The model uses {len(used)} legitimate financial and payment factors "
        "and never uses race, national origin, sex, familial status, disability, "
        "age, or location; protected-class proxies and non-predictive fields are "
        "structurally excluded [1]."
    )
    if parts:
        return " ".join(parts) + ". " + base
    return base


def _deterministic_general() -> str:
    return (
        "I'm the risk decision-support assistant. I can explain an applicant's "
        "estimated late-payment probability and the factors behind it, and I can "
        "describe the model's feature policy — what it uses and what it excludes, "
        "and why [1]. Select an applicant to see their estimate, or ask me which "
        "factors the model uses."
    )


# Shown when a what-if names no field we can vary (or only excluded ones): tell
# the reviewer which levers are available rather than re-scoring nothing.
_WHATIF_GUIDANCE = (
    "For a what-if I can vary income, rent, credit score, monthly debt, savings, "
    "or employment length — for example, \"what if their income were $6,000?\" "
    "The saved estimate above is unchanged."
)


def _deterministic_whatif(wf: dict) -> str:
    base = wf["base"]
    whatif = wf["whatif"]
    name = whatif.get("name") or "This applicant"
    applied = wf.get("overrides") or {}
    applied_txt = ", ".join(f"{k.replace('_', ' ')} = {_num(v)}" for k, v in applied.items())
    delta = wf.get("delta_probability", 0.0)
    direction = "lower" if delta < 0 else ("higher" if delta > 0 else "unchanged")
    lead = (
        f"Exploring a what-if for {name} — this is exploratory and does not change the "
        f"saved estimate [1]. With {applied_txt}, the estimated probability of late "
        f"payment moves from {_pct(base['probability'])} ({base['band']} band) to "
        f"{_pct(whatif['probability'])} ({whatif['band']} band), {_signed_pts(delta)} "
        f"({direction}) [1]."
    )
    band = (
        f" That would shift it from the {base['band']} band to the {whatif['band']} band."
        if wf.get("band_changed")
        else f" That keeps it in the {whatif['band']} band."
    )
    changes = ""
    feats = wf.get("features_changed") or []
    if feats:
        changes = " Model inputs that moved: " + "; ".join(c["label"] for c in feats) + " [2]."
    rejected = wf.get("rejected") or {}
    rej_note = ""
    if rejected:
        rej_note = (
            " I couldn't vary "
            + ", ".join(k.replace("_", " ") for k in rejected)
            + " — those aren't model inputs (they're excluded)."
        )
    guard = (
        " This is decision-support only and hypothetical: the saved estimate is "
        "unchanged, and an elevated estimate simply routes the application to a person "
        "for review."
    )
    return lead + band + changes + rej_note + guard


def _deterministic_counterfactual(cf: dict) -> str:
    base = cf["base"]
    final = cf["final"]
    name = base.get("name") or "This applicant"
    tb = cf.get("target_band", "medium")
    steps = cf.get("steps") or []
    lead = (
        f"Exploring what the model projects could move {name} toward the {tb} band — this "
        f"is exploratory and illustrative, not a recommendation and not a change to the "
        f"saved estimate [1]. The saved estimate is {_pct(base['probability'])} "
        f"({base['band']} band) [1]."
    )
    if steps:
        body = " The modeled changes with the largest impact, applied in order: " + "; ".join(
            f"{s['label']}, bringing the estimate to {_pct(s['probability_after'])} "
            f"({s['band_after']} band)"
            for s in steps
        ) + " [2]."
    else:
        body = (
            " None of the modeled financial levers lower the estimate within realistic "
            "bounds [2]."
        )
    if cf.get("achievable"):
        outcome = f" Together these reach the {tb} band ({_pct(final['probability'])})."
    else:
        outcome = (
            f" Even together these do not reach the {tb} band; the projected estimate is "
            f"{_pct(final['probability'])} ({final['band']} band)."
        )
    guard = (
        " These are hypothetical illustrations using only the model's financial inputs; "
        "the saved estimate is unchanged, and elevated estimates route to a person for "
        "review."
    )
    return lead + body + outcome + guard


def _deterministic_compare(cmp: dict) -> str:
    result = cmp["result"]
    ps = cmp["portfolio"]
    name = result.get("name") or "This applicant"
    lead = (
        f"{name}'s estimated probability of late payment is {_pct(result['probability'])} "
        f"({result['band']} band) [1]. Across {ps.get('scored', 0)} scored applicants the "
        f"portfolio average is {_pct(ps.get('avg_probability'))}, so this applicant is "
        f"{_signed_pts(cmp.get('vs_avg'))} versus the average and sits at about the "
        f"{cmp.get('percentile')}th percentile (rank {cmp.get('band_rank')} by estimated "
        f"risk, where 1 is highest) [1]."
    )
    bc = ps.get("band_counts") or {}
    dist = (
        f" For context, the portfolio splits into {bc.get('low', 0)} low, "
        f"{bc.get('medium', 0)} medium, and {bc.get('high', 0)} high [2]."
    )
    guard = (
        " This is neutral context for a human reviewer — a relative position, not a "
        "leasing decision about anyone."
    )
    return lead + dist + guard


def _deterministic_portfolio(ps: dict) -> str:
    bc = ps.get("band_counts") or {}
    return (
        f"Across {ps.get('scored', 0)} scored applicants the average estimated probability "
        f"of late payment is {_pct(ps.get('avg_probability'))} and the median is "
        f"{_pct(ps.get('median_probability'))}; the high-risk share is "
        f"{ps.get('high_risk_pct')}% [1]. The bands split into {bc.get('low', 0)} low, "
        f"{bc.get('medium', 0)} medium, and {bc.get('high', 0)} high [1]. Select an applicant "
        f"to compare one against this distribution. This is decision-support context for a "
        f"human reviewer, not a leasing decision."
    )


# ---------------------------------------------------------------------------
# LLM synthesis (optional; None offline or on any error)
# ---------------------------------------------------------------------------
def _coalesce(content) -> str:
    """LangChain chunk/message ``.content`` can be a str or a list of blocks."""
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                parts.append(str(b.get("text", "")))
            else:
                parts.append(str(b))
        return "".join(parts)
    return str(content or "")


def _build_messages(question, context_blocks, history, system):
    numbered = "\n\n".join(
        f"[{i+1}] {block}" for i, block in enumerate(context_blocks)
    )
    messages = [("system", system)]
    for turn in (history or [])[-6:]:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role in ("user", "human") and content:
            messages.append(("human", content))
        elif role in ("assistant", "ai", "bot") and content:
            messages.append(("ai", content))
    messages.append(
        (
            "human",
            f"Context:\n{numbered}\n\nQuestion: {question}\n\n"
            f"Answer using only the context above, with [n] citations.",
        )
    )
    return messages


def _llm_answer(question, context_blocks, history, system=_RISK_SYSTEM) -> str | None:
    llm = get_langchain_llm()
    if llm is None:
        return None
    try:
        messages = _build_messages(question, context_blocks, history, system)
        raw = llm.invoke(messages).content
        text = _coalesce(raw).strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 — degrade to deterministic
        print(f"risk_chat: LLM synthesis failed ({type(exc).__name__}); templated.")
        return None


# ---------------------------------------------------------------------------
# TARGET-BAND parsing for counterfactuals
# ---------------------------------------------------------------------------
_TARGET_BAND_RE = re.compile(r"\b(low|medium|moderate|high)\b", re.IGNORECASE)
# When no explicit target is named ("move them to a lower band"), aim one band
# down from where they are now (counterfactuals only ever lower risk).
_NEXT_LOWER_BAND = {"high": "medium", "medium": "low", "low": "low"}


def _parse_target_band(question: str, current_band: str | None) -> str:
    """The band a counterfactual should aim for. An explicit low/medium wins; a
    named 'high' (or nothing) falls back to one band below the current band."""
    m = _TARGET_BAND_RE.search(question or "")
    if m:
        word = m.group(1).lower()
        if word == "moderate":
            return "medium"
        if word in ("low", "medium"):
            return word
    return _NEXT_LOWER_BAND.get(current_band or "medium", "low")


# ---------------------------------------------------------------------------
# SHARED DETERMINISTIC PASS (mirror concierge._Plan)
# ---------------------------------------------------------------------------
class _RiskPlan:
    """The fast, deterministic groundwork shared by the (future) streaming and
    the non-streaming paths: routing, tool execution, numbered context + sources,
    the artifact, follow-ups, the system prompt, and a templated fallback.

    The what-if / counterfactual / compare intents run their own ``risk_agent``
    tools here (Phase 2); a tool hiccup degrades to the grounded explain view
    rather than failing."""

    __slots__ = (
        "intent", "scope", "result", "gov", "note",
        "whatif", "cf", "cmp", "ps",
        "context_blocks", "sources", "artifact", "follow_ups", "system",
    )

    def __init__(self, question, applicant_id):
        q = question or ""
        self.intent = route(q, applicant_id)
        self.scope = "applicant" if applicant_id else "portfolio"
        self.note = ""
        self.gov: dict | None = None
        self.whatif: dict | None = None
        self.cf: dict | None = None
        self.cmp: dict | None = None
        self.ps: dict | None = None

        # Load the applicant's estimate up front (None if unknown/unscoped).
        explained = risk_agent.explain_reasons(applicant_id)
        self.result: dict | None = explained["result"] if explained else None

        if self.intent == "exclusions":
            self.gov = risk_agent.model_governance(q)
            self.context_blocks, self.sources = _governance_context(self.gov)
            self.artifact = risk_agent._none_artifact()
            self.follow_ups = _follow_ups("exclusions")
        elif self.intent == "whatif" and self.result is not None:
            self._plan_whatif(q, applicant_id)
        elif self.intent == "counterfactual" and self.result is not None:
            self._plan_counterfactual(q, applicant_id)
        elif self.intent == "compare":
            self._plan_compare(applicant_id)
        elif self.result is not None:
            # explain (default applicant intent).
            self.context_blocks, self.sources = _explain_context(self.result)
            self.artifact = risk_agent._score_artifact(self.result)
            self.follow_ups = _follow_ups("explain")
        elif self.intent == "general":
            self.context_blocks, self.sources = _general_context()
            self.artifact = risk_agent._none_artifact()
            self.follow_ups = _follow_ups("general")
        else:
            # An applicant intent with no resolvable estimate — deflect
            # gracefully. Empty context => deterministic deflection.
            self.context_blocks, self.sources = [], []
            self.artifact = risk_agent._none_artifact()
            self.follow_ups = _follow_ups("general")

        self.system = _system_for(self.intent)

    # -- Phase-2 intent planners (each degrades to explain on any tool error) --
    def _plan_explain_fallback(self) -> None:
        """Ground on the current saved estimate — the safe view when a Phase-2
        tool cannot produce its own artifact."""
        self.context_blocks, self.sources = _explain_context(self.result)
        self.artifact = risk_agent._score_artifact(self.result)

    def _plan_whatif(self, q: str, applicant_id) -> None:
        self.follow_ups = _follow_ups("whatif")
        try:
            overrides = risk_agent._parse_overrides(q)
            wf = risk_agent.score_whatif(applicant_id, overrides)
        except Exception as exc:  # noqa: BLE001 — degrade to explain
            print(f"risk_chat: score_whatif failed ({type(exc).__name__}: {exc}).")
            self._plan_explain_fallback()
            return
        if not wf or not wf.get("overrides"):
            # Nothing we can vary (no parseable lever or only excluded keys):
            # keep the saved estimate on screen and explain what CAN be varied.
            self._plan_explain_fallback()
            rejected = (wf or {}).get("rejected") or {}
            if rejected:
                self.note = (
                    "I couldn't vary "
                    + ", ".join(k.replace("_", " ") for k in rejected)
                    + " — those aren't model inputs. " + _WHATIF_GUIDANCE
                )
            else:
                self.note = _WHATIF_GUIDANCE
            return
        self.whatif = wf
        self.context_blocks, self.sources = _whatif_context(wf)
        changes = [
            {"feature": c["feature"], "label": c["label"], "from": c["from"], "to": c["to"]}
            for c in wf.get("features_changed", [])
        ]
        self.artifact = risk_agent._whatif_artifact(
            result=wf["whatif"], baseline=wf["base"]["probability"], changes=changes
        )

    def _plan_counterfactual(self, q: str, applicant_id) -> None:
        self.follow_ups = _follow_ups("counterfactual")
        try:
            target = _parse_target_band(q, self.result.get("band"))
            cf = risk_agent.counterfactual(applicant_id, target)
        except Exception as exc:  # noqa: BLE001 — degrade to explain
            print(f"risk_chat: counterfactual failed ({type(exc).__name__}: {exc}).")
            self._plan_explain_fallback()
            return
        if not cf:
            self._plan_explain_fallback()
            return
        self.cf = cf
        self.context_blocks, self.sources = _counterfactual_context(cf)
        changes = [
            {"feature": s["feature"], "label": s["label"], "from": s["from"], "to": s["to"]}
            for s in cf.get("steps", [])
        ]
        self.artifact = risk_agent._counterfactual_artifact(
            target_band=cf["target_band"],
            achievable=cf["achievable"],
            changes=changes,
            result=cf.get("final"),
        )

    def _plan_compare(self, applicant_id) -> None:
        self.follow_ups = _follow_ups("compare")
        # Applicant scope: position this one against the portfolio.
        if applicant_id and self.result is not None:
            try:
                cmp = risk_agent.compare(applicant_id)
            except Exception as exc:  # noqa: BLE001 — degrade to explain
                print(f"risk_chat: compare failed ({type(exc).__name__}: {exc}).")
                self._plan_explain_fallback()
                return
            if cmp:
                self.cmp = cmp
                self.context_blocks, self.sources = _compare_context(cmp)
                subject = {
                    "label": cmp["result"].get("name") or "This applicant",
                    "probability": cmp["result"]["probability"],
                    "band": cmp["result"]["band"],
                    "applicant_id": cmp["result"].get("applicant_id", ""),
                }
                self.artifact = risk_agent._comparison_artifact(
                    subject=subject,
                    rows=_portfolio_rows(cmp["portfolio"]),
                    percentile=cmp["percentile"],
                )
                return
            self._plan_explain_fallback()
            return
        # Portfolio scope (no applicant selected): summarize the distribution.
        try:
            ps = risk_agent.portfolio_summary()
        except Exception as exc:  # noqa: BLE001
            print(f"risk_chat: portfolio_summary failed ({type(exc).__name__}: {exc}).")
            ps = None
        if not ps or not ps.get("scored"):
            self.context_blocks, self.sources = _general_context()
            self.artifact = risk_agent._none_artifact()
            self.follow_ups = _follow_ups("general")
            return
        self.ps = ps
        self.context_blocks, self.sources = _portfolio_context(ps)
        avg = ps.get("avg_probability", 0.0)
        self.artifact = risk_agent._comparison_artifact(
            subject={"label": "Portfolio average", "probability": avg, "band": _band_of(avg)},
            rows=_portfolio_rows(ps),
        )

    def deterministic_answer(self) -> str:
        if self.intent == "exclusions" and self.gov is not None:
            return _deterministic_exclusions(self.gov)
        if self.whatif is not None:
            return _deterministic_whatif(self.whatif)
        if self.cf is not None:
            return _deterministic_counterfactual(self.cf)
        if self.cmp is not None:
            return _deterministic_compare(self.cmp)
        if self.ps is not None:
            return _deterministic_portfolio(self.ps)
        if self.result is not None:
            return _deterministic_explain(self.result, self.note)
        if self.intent == "general":
            return _deterministic_general()
        return _DEFLECTION


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT — never raises
# ---------------------------------------------------------------------------
def answer(question: str, applicant_id: str | None = None, history=None) -> dict:
    try:
        return _answer(question, applicant_id, history)
    except Exception as exc:  # noqa: BLE001 — mirror concierge.answer: never 500
        print(f"risk_chat: unexpected error ({type(exc).__name__}: {exc}); safe fallback.")
        try:
            intent = route(question or "", applicant_id)
        except Exception:  # noqa: BLE001
            intent = "general"
        return {
            "answer": (
                "Sorry, I hit a snag looking that up. Please try rephrasing, or "
                "select an applicant to see their risk estimate."
            ),
            "scope": "applicant" if applicant_id else "portfolio",
            "applicant_id": applicant_id or "",
            "intent": intent,
            "sources": [],
            "artifact": {"kind": "none"},
            "follow_ups": _follow_ups(intent),
            "source": "rules",
        }


def _answer(question: str, applicant_id: str | None, history) -> dict:
    plan = _RiskPlan(question, applicant_id)

    llm_text = (
        _llm_answer(question, plan.context_blocks, history, plan.system)
        if plan.context_blocks
        else None
    )
    if llm_text is not None:
        source = "anthropic"
        text = llm_text
    else:
        source = "rules"
        text = plan.deterministic_answer()

    return {
        "answer": text,
        "scope": plan.scope,
        "applicant_id": applicant_id or "",
        "intent": plan.intent,
        "sources": plan.sources,
        "artifact": plan.artifact,
        "follow_ups": plan.follow_ups,
        "source": source,
    }
