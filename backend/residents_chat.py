"""Residents Chat — the decision-support agent for the Residents page.

``answer(question, resident_id=None, property_id=None, history=None) -> dict``
mirrors ``risk_chat.answer`` EXACTLY:

  1. ROUTER   — ``route`` deterministically classifies the question into one of
                explain | horizon | frequency | severity | arrears | cure |
                retention | property_health | compare | governance | general
                (cheap regexes, precedence-ordered).
  2. TOOLS    — thin deterministic wrappers over ``residents_risk`` (score a
                resident, pull specific heads by family/horizon, property /
                portfolio health, model-card governance) return EXACT structured
                numbers. EVERY number the agent states originates in a head
                payload — the LLM never computes or invents a probability.
  3. SYNTHESIZE — Claude (via ``llm.get_langchain_llm``) rewrites ONLY the
                assembled numbered context into prose with inline [n] citations
                (source="anthropic"). Offline / on any error we emit a templated
                grounded answer (source="rules").

DECISION-SUPPORT ONLY: the answers plan proactive outreach and retention. They
NEVER evict, deny, price, condition a lease, or take an automated action; a
serious-delinquency estimate always "routes to a person". Like
``risk_chat.answer`` this NEVER raises — the outermost guard always returns a
safe response dict.
"""

from __future__ import annotations

import re

import graph
import residents_risk
from llm import get_langchain_llm


# ---------------------------------------------------------------------------
# ROUTER — intent classification (precedence-ordered; cheap regexes)
# ---------------------------------------------------------------------------
# Governance / feature-policy: "what do you use", "how is it measured",
# "what's excluded", protected-class / fairness questions. Checked FIRST so a
# question naming a feature is treated as governance, not a scored explanation.
_GOVERNANCE_RE = re.compile(
    r"\b("
    r"exclud|not used|don'?t use|do(?:es)? (?:it|the model|you) use|is .* used|"
    r"which features|what features|what factors|what data|what inputs?|"
    r"feature[s]? (?:used|considered)|variables?|"
    r"how (?:is|are|do you|does it) .* (?:measured|calculated|computed|defined|work)|"
    r"how measured|how do you (?:know|decide|predict)|what does the model (?:use|look at)|"
    r"model card|governance|fair|fairness|bias|biased|discriminat|protected|proxy|redlin|"
    r"race|ethnic|national origin|gender|\bsex\b|disab|religion|"
    r"familial|marital|dependents?|children|\bage\b|\bstudent\b|criminal"
    r")\b",
    re.IGNORECASE,
)

# Property / portfolio health: healthiest / worst apartments, this property's
# health, best-to-worst ranking.
_PROPERTY_HEALTH_RE = re.compile(
    r"\b("
    r"healthiest|unhealthiest|health(?:y|iest)? (?:propert|apartment|building|communit)|"
    r"property health|portfolio health|"
    r"(?:which|what) (?:propert|apartment|building|communit|unit)s?|"
    r"best (?:propert|apartment|building|communit)|worst (?:propert|apartment|building|communit)|"
    r"needs attention|rank(?:ing)? .* (?:propert|apartment|building)|"
    r"how (?:healthy|is this property|is my property|are (?:my|the) propert)"
    r")\b",
    re.IGNORECASE,
)

# Compare: this resident vs their property vs the portfolio.
_COMPARE_RE = re.compile(
    r"\b("
    r"compare|versus|\bvs\.?\b|percentile|\brank\b|ranking|"
    r"relative to|compared to|against (?:the )?(?:portfolio|property|others|average)|"
    r"how does .* compare|typical resident|other residents?|the average"
    r")\b",
    re.IGNORECASE,
)

# Cure: will they clear / catch up on their outstanding balance. Checked before
# arrears so "clear their balance" is a cure question, not an arrears-$ one.
_CURE_RE = re.compile(
    r"\b("
    r"cure|clear (?:the|their|his|her|this)? ?balance|clear it|"
    r"catch up|caught up|pay (?:it|the balance|them)? ?off|pay(?:ing)? down|"
    r"get current|become current|bring (?:the )?balance to zero|"
    r"resolve (?:the|their)? ?(?:balance|arrears|debt)|when will .* (?:cured|cleared|current)"
    r")\b",
    re.IGNORECASE,
)

# Retention / renewal: will they renew / stay / move out / churn.
_RETENTION_RE = re.compile(
    r"\b("
    r"renew|renewal|non[\s-]?renew|churn|retain|retention|"
    r"move out|moving out|move-out|leave|leaving|vacat|"
    r"will (?:they|he|she) stay|are they staying|lease end"
    r")\b",
    re.IGNORECASE,
)

# Arrears $: expected balance / arrears / how much owed.
_ARREARS_RE = re.compile(
    r"\b("
    r"arrears|balance|owe|owed|owing|"
    r"how much|expected (?:\$|dollar|amount|debt)|"
    r"\$|dollar|delinquent balance|outstanding|peak balance|behind on"
    r")\b",
    re.IGNORECASE,
)

# Frequency: how often / how many times late / number of late months.
_FREQUENCY_RE = re.compile(
    r"\b("
    r"how often|how many (?:times|months|payments)|number of (?:times|late|missed)|"
    r"how frequent|frequency|how many late|count of late|times late|"
    r"how many .* (?:late|missed)"
    r")\b",
    re.IGNORECASE,
)

# Severity: how bad / worst / days late / 30-60-90 / delinquency bucket.
_SEVERITY_RE = re.compile(
    r"\b("
    r"how (?:bad|severe|serious)|how far behind|worst|severity|serious(?:ly)?|"
    r"days? late|30[\s-]?(?:day|plus|\+)|60[\s-]?(?:day|plus|\+)|90[\s-]?(?:day|plus|\+)|"
    r"delinquen|bucket|escalat|30 days|60 days|90 days"
    r")\b",
    re.IGNORECASE,
)

# Horizon: likelihood of paying late over a window (month / quarter / 6mo / year).
_HORIZON_RE = re.compile(
    r"\b("
    r"likely|likelihood|chance|probab|will .* pay late|will .* be late|"
    r"pay late|paying late|be late|going to be late|risk of (?:being |paying )?late|"
    r"next (?:month|quarter|year|6 months|six months|12 months)|"
    r"this (?:year|quarter|month)|over the next"
    r")\b",
    re.IGNORECASE,
)

# Explain: why is this resident risky / drivers / reasons.
_EXPLAIN_RE = re.compile(
    r"\b("
    r"why|explain|reason|driver|factor|elevated|risky|high[\s-]?risk|"
    r"what'?s driving|what is driving|break ?down|understand|concern|flag"
    r")\b",
    re.IGNORECASE,
)


def route(question: str, resident_id: str | None = None, property_id: str | None = None) -> str:
    """Classify the question into an intent.

    Precedence: governance, property_health, compare, cure, retention, arrears,
    frequency, severity, horizon, explain — then a scope default (``explain``
    when a resident is selected, ``property_health`` when only a property is
    selected, else ``general``)."""
    q = question or ""
    if _GOVERNANCE_RE.search(q):
        return "governance"
    if _PROPERTY_HEALTH_RE.search(q):
        return "property_health"
    if _COMPARE_RE.search(q):
        return "compare"
    if _CURE_RE.search(q):
        return "cure"
    if _RETENTION_RE.search(q):
        return "retention"
    if _ARREARS_RE.search(q):
        return "arrears"
    if _FREQUENCY_RE.search(q):
        return "frequency"
    if _SEVERITY_RE.search(q):
        return "severity"
    if _HORIZON_RE.search(q):
        return "horizon"
    if _EXPLAIN_RE.search(q):
        return "explain"
    if resident_id:
        return "explain"
    if property_id:
        return "property_health"
    return "general"


# ---------------------------------------------------------------------------
# Formatting helpers (grounded numeric strings — all values come from heads)
# ---------------------------------------------------------------------------
def _pct(v) -> str:
    try:
        return f"{round(float(v) * 100)}%"
    except (TypeError, ValueError):
        return "n/a"


def _money(v) -> str:
    try:
        return f"${round(float(v)):,}"
    except (TypeError, ValueError):
        return "n/a"


def _num(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f == int(f) else f"{f:.1f}"


def _snippet(text: str, n: int = 220) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _pct_range(head: dict) -> str:
    rng = (head or {}).get("range") or []
    if len(rng) == 2:
        return f"{_pct(rng[0])}–{_pct(rng[1])}"
    return "n/a"


def _money_interval(head: dict) -> str:
    iv = (head or {}).get("interval") or []
    if len(iv) == 2:
        return f"{_money(iv[0])}–{_money(iv[1])}"
    return "n/a"


def _count_interval(head: dict) -> str:
    iv = (head or {}).get("interval") or []
    if len(iv) == 2:
        return f"{_num(iv[0])}–{_num(iv[1])}"
    return "n/a"


def _factors(head: dict) -> tuple[list[str], list[str]]:
    """(increase-labels, decrease-labels) from a head's own reason codes."""
    codes = (head or {}).get("reason_codes", [])
    ups = [c.get("label", "") for c in codes if c.get("direction") == "increases"]
    downs = [c.get("label", "") for c in codes if c.get("direction") == "decreases"]
    return [l for l in ups if l], [l for l in downs if l]


def _low_conf(head: dict) -> bool:
    return (head or {}).get("confidence") == "low"


# ---------------------------------------------------------------------------
# Property-name resolution (display only; never a model input)
# ---------------------------------------------------------------------------
def _property_names() -> dict:
    names: dict = {}
    try:
        for p in graph.load_properties():
            pid = p.get("id")
            if pid:
                names[pid] = p.get("name") or pid
    except Exception:  # noqa: BLE001 — names are cosmetic; fall back to the id
        pass
    return names


def _property_name(property_id: str) -> str:
    if not property_id:
        return ""
    return _property_names().get(property_id, property_id)


# ---------------------------------------------------------------------------
# TOOLS — deterministic wrappers over residents_risk (NO LLM). Return None for
# an unknown resident so callers deflect gracefully.
# ---------------------------------------------------------------------------
def score_resident(resident_id: str | None) -> dict | None:
    """Full multi-head predictions for one resident, or None if unknown. Never
    raises — ``predict_resident`` degrades to heuristics server-side."""
    if not resident_id:
        return None
    try:
        r = residents_risk.get_resident(resident_id)
    except Exception as exc:  # noqa: BLE001
        print(f"residents_chat: get_resident failed ({type(exc).__name__}: {exc}).")
        return None
    if r is None:
        return None
    try:
        return residents_risk.predict_resident(r)
    except Exception as exc:  # noqa: BLE001 — predict never raises, but belt-and-braces
        print(f"residents_chat: predict_resident failed ({type(exc).__name__}: {exc}).")
        return None


def property_health(property_id: str | None) -> dict | None:
    if not property_id:
        return None
    try:
        h = residents_risk.property_health(property_id)
    except Exception as exc:  # noqa: BLE001
        print(f"residents_chat: property_health failed ({type(exc).__name__}: {exc}).")
        return None
    if h:
        h = dict(h)
        h["name"] = _property_name(property_id)
    return h


def portfolio_health() -> list:
    try:
        ranked = residents_risk.portfolio_health() or []
    except Exception as exc:  # noqa: BLE001
        print(f"residents_chat: portfolio_health failed ({type(exc).__name__}: {exc}).")
        return []
    names = _property_names()
    out = []
    for h in ranked:
        h = dict(h)
        h["name"] = names.get(h.get("property_id"), h.get("property_id") or "")
        out.append(h)
    return out


# ---------------------------------------------------------------------------
# GOVERNANCE — feature-policy lookup for the residents model.
# ---------------------------------------------------------------------------
_USED_SYNONYMS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"autopay|auto[\s-]?pay|automatic payment", re.I), "autopay_enrolled",
     "Whether the resident is enrolled in autopay — a legitimate payment-behavior factor."),
    (re.compile(r"late payment|payment history|paid late|lateness|track record", re.I), "late_count_12mo",
     "Prior late-payment counts from the rent ledger — a legitimate payment-history factor."),
    (re.compile(r"days? late|how late|severity", re.I), "max_days_late_12mo",
     "Worst days-late reached in the ledger — a legitimate payment-severity factor."),
    (re.compile(r"balance|arrears|owe|outstanding", re.I), "current_balance",
     "Current outstanding balance from the ledger — a legitimate collections factor."),
    (re.compile(r"rent[\s-]?to[\s-]?income|rent burden|burden|affordab", re.I), "rent_to_income",
     "Rent as a share of income — a legitimate affordability factor."),
    (re.compile(r"tenure|how long|length of tenancy|time (?:as|been) a? ?resident", re.I), "tenure_months",
     "Months of tenancy — a legitimate tenancy factor."),
    (re.compile(r"income verif|verified income|income on file", re.I), "income_verified",
     "Whether income on file is verified — a legitimate data-quality factor."),
    (re.compile(r"notice|notices sent|notice response", re.I), "notices_sent_12mo",
     "Notices sent / responded to — a legitimate communication factor."),
    (re.compile(r"lease end|renewal timing|months to lease", re.I), "months_to_lease_end",
     "Months until lease end — a legitimate lease-timing factor (retention heads)."),
    (re.compile(r"prior renewal|renewed before|renewal history", re.I), "prior_renewals",
     "Prior lease renewals — a legitimate tenancy factor (retention heads)."),
]

_EXCLUDED_SYNONYMS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\brace\b|ethnicit|national origin", re.I), "race / national origin",
     "Protected class under fair-housing law; never collected or used."),
    (re.compile(r"\bsex\b|gender", re.I), "sex / gender",
     "Protected class under fair-housing law; never collected or used."),
    (re.compile(r"disab", re.I), "disability",
     "Protected class under fair-housing law; never collected or used."),
    (re.compile(r"religion|faith", re.I), "religion",
     "Protected class under fair-housing law; never collected or used."),
    (re.compile(r"familial|marital|married|household size|household", re.I), "familial / household status",
     "Familial status — protected under fair-housing law."),
    (re.compile(r"dependent|\bchildren\b|\bkids\b", re.I), "dependents",
     "Familial status — protected under fair-housing law."),
    (re.compile(r"\bage\b|birth ?date|date of birth|\bdob\b|is_?student|\bstudent\b", re.I), "age / student status",
     "Age proxy — protected; excluded from every model."),
    (re.compile(r"\blocation\b|neighborhood|\bcity\b|\bzip\b|redlin|where (?:they|you) live|address|property id", re.I),
     "location / property / neighborhood",
     "Location proxy for protected classes (redlining risk); kept only to audit fairness, never a model input."),
    (re.compile(r"\bname\b|unit id|identity", re.I), "name / unit id",
     "Identity / display only; not predictive and never a model input."),
    (re.compile(r"criminal|conviction|\bfelony\b|\bcrime\b", re.I), "criminal record",
     "Disparate-impact risk; not collected or used."),
]


def model_governance(query: str = "") -> dict:
    """The model card plus a ``feature_lookup`` resolving any field(s) named in
    the query to used (a legitimate ledger/tenancy factor) or excluded (with its
    documented reason). Never raises."""
    q = query or ""
    lookup: list[dict] = []
    seen: set[str] = set()
    for rx, feat, reason in _USED_SYNONYMS:
        if feat in seen:
            continue
        if rx.search(q):
            lookup.append({"field": feat, "status": "used", "reason": reason})
            seen.add(feat)
    for rx, field, reason in _EXCLUDED_SYNONYMS:
        if field in seen:
            continue
        if rx.search(q):
            lookup.append({"field": field, "status": "excluded", "reason": reason})
            seen.add(field)
    try:
        card = residents_risk.model_card()
    except Exception:  # noqa: BLE001
        card = {}
    return {"card": card, "feature_lookup": lookup}


# ---------------------------------------------------------------------------
# HORIZON parsing (which late-family head the question is asking about)
# ---------------------------------------------------------------------------
def _parse_horizon(question: str) -> str:
    q = (question or "").lower()
    if re.search(r"\bnext month\b|\ba month\b|30 day|monthly|this month", q):
        return "late_1m"
    if re.search(r"\bquarter\b|3 month|three month|90 day|next 3", q):
        return "late_3m"
    if re.search(r"6 month|six month|180 day|half a year|half-year", q):
        return "late_6m"
    if re.search(r"\byear\b|12 month|twelve month|annual|next 12", q):
        return "late_12m"
    return "late_12m"  # "next year" is the headline horizon


# ---------------------------------------------------------------------------
# ARTIFACT builders (discriminated union on ``kind``)
# ---------------------------------------------------------------------------
def _none_artifact() -> dict:
    return {"kind": "none"}


def _resident_artifact(pred: dict, head_names: list, intent: str) -> dict:
    """The relevant head payloads for one resident — carries the EXACT numbers the
    answer cites so the UI can render gauges/tiles and grounding is provable."""
    heads = pred.get("heads") or {}
    selected = {n: heads[n] for n in head_names if n in heads}
    return {
        "kind": "resident",
        "intent": intent,
        "resident_id": pred.get("resident_id", ""),
        "name": pred.get("name", ""),
        "property_id": pred.get("property_id", ""),
        "property_name": _property_name(pred.get("property_id", "")),
        "heads": selected,
    }


def _portfolio_health_artifact(ranked: list) -> dict:
    return {
        "kind": "property_health",
        "scope": "portfolio",
        "properties": ranked,
        "healthiest": ranked[0] if ranked else None,
        "needs_attention": ranked[-1] if ranked else None,
        "count": len(ranked),
    }


def _property_health_artifact(health: dict, ranked: list | None = None) -> dict:
    art = {"kind": "property_health", "scope": "property", "property": health}
    if ranked:
        art["properties"] = ranked
    return art


def _compare_artifact(pred: dict, health: dict | None, ranked: list) -> dict:
    heads = pred.get("heads") or {}
    return {
        "kind": "compare",
        "resident_id": pred.get("resident_id", ""),
        "name": pred.get("name", ""),
        "property_id": pred.get("property_id", ""),
        "property_name": _property_name(pred.get("property_id", "")),
        "heads": {n: heads[n] for n in ("late_12m", "serious") if n in heads},
        "property": health,
        "portfolio": ranked,
    }


# ---------------------------------------------------------------------------
# HEAD SELECTION per intent (which heads carry the numbers for the answer)
# ---------------------------------------------------------------------------
_HEADS_FREQUENCY = ["late_count_12m", "missed_count_12m"]
_HEADS_SEVERITY = ["max_days_late_12m", "p_30d_12m", "p_60d_12m", "p_90d_12m",
                   "delinquency_bucket_12m", "serious"]
_HEADS_ARREARS = ["arrears_3m", "arrears_12m", "peak_balance_12m"]
_HEADS_CURE = ["p_cure_6m", "months_to_cure"]
_HEADS_RETENTION = ["churn", "churn_12m"]
_HEADS_LATE = ["late_1m", "late_3m", "late_6m", "late_12m"]
_HEADS_EXPLAIN = ["late_12m", "late_3m", "serious"]


# ---------------------------------------------------------------------------
# FOLLOW-UP SUGGESTIONS (deterministic — work with the LLM offline)
# ---------------------------------------------------------------------------
_FOLLOWUPS = {
    "explain": [
        "How likely are they to pay late over the next year?",
        "What's their expected balance next year?",
        "What does the model use to measure this?",
    ],
    "horizon": [
        "How often will they pay late next year?",
        "How severe could it get?",
        "Why is their risk at this level?",
    ],
    "frequency": [
        "How severe could the lateness get?",
        "What's their expected balance next year?",
        "Why is their risk at this level?",
    ],
    "severity": [
        "How likely are they to pay late next year?",
        "Will they clear their balance?",
        "Why is their risk at this level?",
    ],
    "arrears": [
        "Will they clear their balance?",
        "How often will they pay late?",
        "Why is their risk at this level?",
    ],
    "cure": [
        "What's their expected balance next year?",
        "How likely are they to pay late again?",
        "Why is their risk at this level?",
    ],
    "retention": [
        "Why might they not renew?",
        "How likely are they to pay late?",
        "What does the model use to measure this?",
    ],
    "property_health": [
        "Which property needs the most attention?",
        "What's dragging the lowest-scoring property down?",
        "What does the health score measure?",
    ],
    "compare": [
        "Why is this resident's risk at this level?",
        "How does their property compare to the portfolio?",
        "How likely are they to pay late next year?",
    ],
    "governance": [
        "Is autopay used by the model?",
        "Is location or age ever used?",
        "How is the risk measured?",
    ],
    "general": [
        "Which properties are healthiest?",
        "What does the model use to measure risk?",
        "Which property needs the most attention?",
    ],
}


def _follow_ups(intent: str) -> list[str]:
    return list(_FOLLOWUPS.get(intent, _FOLLOWUPS["general"]))


# ---------------------------------------------------------------------------
# SYSTEM PROMPTS + guardrails
# ---------------------------------------------------------------------------
_RESIDENT_SYSTEM = (
    "You are the Resident Risk assistant — DECISION-SUPPORT ONLY, for planning "
    "PROACTIVE OUTREACH and RETENTION for CURRENT residents. Answer using ONLY "
    "the numbered context provided; never invent or recompute a probability, "
    "count, dollar amount, band, or reason code — every number must come from "
    "the context. Cite the context you use inline with bracketed numbers like "
    "[1] or [2]. Explain risk through the model's OWN reason codes. You MUST "
    "NEVER recommend or imply eviction, denial, non-renewal, pricing, late fees, "
    "lease conditioning, or any automated action, and never say a resident "
    "should be removed or penalized — a serious-delinquency estimate simply "
    "routes to a person for a supportive, human review. Frame elevated risk as a "
    "prompt for outreach (payment plans, autopay, check-ins), not punishment. "
    "Refuse to use or speculate about protected attributes (race, national "
    "origin, sex, familial status, disability, age, religion, location) and cite "
    "the model card's reason. Note low confidence honestly when a head is flagged "
    "(rare 60+/90+ day events, or long-horizon estimates for short-tenure "
    "residents). The model is trained on SYNTHETIC data; results are estimates, "
    "not guarantees, and not a consumer report. Be concise and factual."
)

_GOVERNANCE_SYSTEM = (
    "You are the Resident Risk assistant explaining the model's FEATURE POLICY / "
    "governance and how outcomes are measured. Answer using ONLY the numbered "
    "context; never invent features, reasons, or metrics. Cite inline with [n]. "
    "State plainly which fields are used (legitimate ledger / tenancy / "
    "affordability factors) and which are structurally excluded (protected-class "
    "proxies or non-predictive fields), and give the model card's reason. Never "
    "recommend eviction, denial, pricing, or any automated action. This is "
    "decision-support on synthetic data — an estimate, not a consumer report. Be "
    "concise and factual."
)

_HEALTH_SYSTEM = (
    _RESIDENT_SYSTEM
    + " This turn ranks properties by a composite HEALTH score (0-100, higher = "
    "healthier) built only from resident predictions. Report only the scores, "
    "grades, and drivers in the context as neutral context for a regional "
    "director planning where to focus outreach — never as a ranking that "
    "penalizes a property or anyone in it."
)

_COMPARE_SYSTEM = (
    _RESIDENT_SYSTEM
    + " This turn positions one resident against their property and the portfolio. "
    "Report only the relative position in the context as neutral context for a "
    "human planning outreach, never as a ranking that penalizes anyone."
)


def _system_for(intent: str) -> str:
    if intent == "governance":
        return _GOVERNANCE_SYSTEM
    if intent == "property_health":
        return _HEALTH_SYSTEM
    if intent == "compare":
        return _COMPARE_SYSTEM
    return _RESIDENT_SYSTEM


# ---------------------------------------------------------------------------
# DEFLECTIONS
# ---------------------------------------------------------------------------
_DEFLECTION = (
    "I don't have predictions for that resident. Select a resident from the list "
    "to see their late-payment, arrears, cure, and renewal estimates and the "
    "factors behind them, or ask me which properties are healthiest or what the "
    "model uses."
)

_GENERAL = (
    "I'm the residents decision-support assistant. For a selected resident I can "
    "explain how likely they are to pay late (next month through next year), how "
    "often and how severe it might get, their expected balance and whether it's "
    "likely to clear, and their renewal risk — all from the model's own signals, "
    "for planning proactive outreach [1]. I can also rank your properties by a "
    "composite health score, and describe what the model uses and what it "
    "excludes. Select a resident, or ask which properties are healthiest."
)


# ---------------------------------------------------------------------------
# CONTEXT + DETERMINISTIC ANSWER per intent. Context blocks are numbered; the
# deterministic answer cites the same [n]. Numbers come only from heads.
# ---------------------------------------------------------------------------
def _guard_outreach() -> str:
    return (
        " This is decision-support for proactive outreach only — an elevated "
        "estimate is a prompt to reach out (a payment plan, autopay, a check-in), "
        "not eviction, non-renewal, or any automated action; serious cases route "
        "to a person."
    )


def _conf_note(*heads: dict) -> str:
    if any(_low_conf(h) for h in heads if h):
        return (
            " Confidence is lower here (thin history / short tenure, or a rare "
            "event) — treat it as directional."
        )
    return ""


class _ResidentPlan:
    """The fast, deterministic groundwork shared by the streaming and
    non-streaming paths: routing, tool execution, numbered context + sources,
    the artifact (the relevant head payloads or the property-health ranking),
    follow-ups, the system prompt, and a templated fallback. Mirrors
    ``risk_chat._RiskPlan``; a tool hiccup degrades gracefully rather than
    failing."""

    __slots__ = (
        "intent", "scope", "resident_id", "property_id",
        "pred", "gov", "health", "ranked",
        "context_blocks", "sources", "artifact", "follow_ups", "system",
    )

    def __init__(self, question, resident_id, property_id):
        q = question or ""
        self.intent = route(q, resident_id, property_id)
        self.resident_id = resident_id or ""
        self.property_id = property_id or ""
        self.scope = "resident" if resident_id else ("property" if property_id else "portfolio")
        self.pred: dict | None = None
        self.gov: dict | None = None
        self.health: dict | None = None
        self.ranked: list = []
        self.context_blocks: list = []
        self.sources: list = []
        self.artifact = _none_artifact()
        self.follow_ups = _follow_ups(self.intent)

        if self.intent == "governance":
            self._plan_governance(q)
        elif self.intent == "property_health":
            self._plan_property_health()
        elif self.intent == "general" and not resident_id:
            self._plan_general()
        elif self.intent == "compare":
            self._plan_compare(resident_id)
        else:
            # A resident-scoped intent. Score up front; deflect if unknown.
            self.pred = score_resident(resident_id)
            if self.pred is None:
                # No resolvable resident — empty context => deterministic deflection.
                if property_id:
                    self._plan_property_health()
                else:
                    self.context_blocks, self.sources = [], []
                    self.artifact = _none_artifact()
                    self.follow_ups = _follow_ups("general")
            else:
                self._plan_resident(q)

        self.system = _system_for(self.intent)

    # -- resident-scoped planners --------------------------------------------
    def _plan_resident(self, q: str) -> None:
        intent = self.intent
        if intent == "horizon":
            self._plan_horizon(q)
        elif intent == "frequency":
            self._plan_frequency()
        elif intent == "severity":
            self._plan_severity()
        elif intent == "arrears":
            self._plan_arrears()
        elif intent == "cure":
            self._plan_cure()
        elif intent == "retention":
            self._plan_retention()
        else:
            self.intent = "explain"
            self._plan_explain()

    def _name(self) -> str:
        return (self.pred or {}).get("name") or "This resident"

    def _heads(self) -> dict:
        return (self.pred or {}).get("heads") or {}

    def _src(self, label: str, snippet: str) -> dict:
        return {"type": "resident", "label": label, "snippet": _snippet(snippet),
                "resident_id": self.resident_id}

    def _plan_explain(self) -> None:
        heads = self._heads()
        name = self._name()
        late12 = heads.get("late_12m", {})
        late3 = heads.get("late_3m", {})
        serious = heads.get("serious", {})
        ups, downs = _factors(late12) if late12.get("reason_codes") else _factors(late3)
        risk_line = (
            f"Risk snapshot — {name}: over the next year the estimated chance of paying "
            f"late at least once is {_pct(late12.get('probability'))} ({late12.get('band')} "
            f"band, range {_pct_range(late12)}); next quarter it is {_pct(late3.get('probability'))} "
            f"({late3.get('band')} band). Serious-delinquency risk next quarter is "
            f"{_pct(serious.get('probability'))} ({serious.get('band')} band), which routes to a "
            f"human reviewer.{_conf_note(late12, serious)}"
        )
        factor_line = (
            "Key factors (the model's own reason codes). "
            + (f"Raise the estimate: {'; '.join(ups)}. " if ups else "Raise the estimate: none notable. ")
            + (f"Lower the estimate: {'; '.join(downs)}." if downs else "Lower the estimate: none notable.")
        )
        self.context_blocks = [
            f"Risk snapshot — {name}:\n{risk_line}",
            f"Key factors — {name}:\n{factor_line}",
        ]
        self.sources = [
            self._src(f"Risk snapshot — {name}", risk_line),
            self._src(f"Key factors — {name}", factor_line),
        ]
        self.artifact = _resident_artifact(self.pred, _HEADS_EXPLAIN, "explain")
        self.follow_ups = _follow_ups("explain")

    def _plan_horizon(self, q: str) -> None:
        heads = self._heads()
        name = self._name()
        asked = _parse_horizon(q)
        labels = {"late_1m": "next month", "late_3m": "next quarter",
                  "late_6m": "next 6 months", "late_12m": "next year"}
        parts = []
        for hn in _HEADS_LATE:
            h = heads.get(hn, {})
            parts.append(f"{labels[hn]} {_pct(h.get('probability'))} ({h.get('band')} band)")
        asked_h = heads.get(asked, {})
        line = (
            f"Late-payment likelihood — {name} (chance of paying late at least once). "
            + "; ".join(parts) + ". "
            f"For the asked horizon ({labels[asked]}) it is {_pct(asked_h.get('probability'))} "
            f"({asked_h.get('band')} band, range {_pct_range(asked_h)}).{_conf_note(asked_h)}"
        )
        self.context_blocks = [f"Late-payment likelihood — {name}:\n{line}"]
        self.sources = [self._src(f"Late-payment likelihood — {name}", line)]
        art = _resident_artifact(self.pred, _HEADS_LATE, "horizon")
        art["asked_horizon"] = asked
        self.artifact = art
        self.follow_ups = _follow_ups("horizon")

    def _plan_frequency(self) -> None:
        heads = self._heads()
        name = self._name()
        late_c = heads.get("late_count_12m", {})
        missed_c = heads.get("missed_count_12m", {})
        line = (
            f"Late-payment frequency — {name} (next 12 months). Expected late months: "
            f"about {_num(late_c.get('expected'))} (range {_count_interval(late_c)}), of which "
            f"about {_num(missed_c.get('expected'))} fully missed (range {_count_interval(missed_c)})."
            f"{_conf_note(late_c, missed_c)}"
        )
        self.context_blocks = [f"Late-payment frequency — {name}:\n{line}"]
        self.sources = [self._src(f"Late-payment frequency — {name}", line)]
        self.artifact = _resident_artifact(self.pred, _HEADS_FREQUENCY, "frequency")
        self.follow_ups = _follow_ups("frequency")

    def _plan_severity(self) -> None:
        heads = self._heads()
        name = self._name()
        mdl = heads.get("max_days_late_12m", {})
        p30 = heads.get("p_30d_12m", {})
        p60 = heads.get("p_60d_12m", {})
        p90 = heads.get("p_90d_12m", {})
        bucket = heads.get("delinquency_bucket_12m", {})
        line = (
            f"Late-payment severity — {name} (next 12 months). Expected worst days late: "
            f"about {_num(mdl.get('expected'))} days. Chance of reaching 30+ days late "
            f"{_pct(p30.get('probability'))} ({p30.get('band')} band); 60+ days "
            f"{_pct(p60.get('probability'))} (low-power, directional); 90+ days "
            f"{_pct(p90.get('probability'))} (low-power, directional). Most likely worst "
            f"delinquency bucket: {bucket.get('predicted_bucket', 'n/a')}."
            f"{_conf_note(mdl)} The 60+/90+ figures are rare-event estimates — treat as directional."
        )
        self.context_blocks = [f"Late-payment severity — {name}:\n{line}"]
        self.sources = [self._src(f"Late-payment severity — {name}", line)]
        self.artifact = _resident_artifact(self.pred, _HEADS_SEVERITY, "severity")
        self.follow_ups = _follow_ups("severity")

    def _plan_arrears(self) -> None:
        heads = self._heads()
        name = self._name()
        a3 = heads.get("arrears_3m", {})
        a12 = heads.get("arrears_12m", {})
        peak = heads.get("peak_balance_12m", {})
        line = (
            f"Expected arrears — {name}. Projected balance at the end of next quarter "
            f"{_money(a3.get('expected'))} (range {_money_interval(a3)}); at the end of next year "
            f"{_money(a12.get('expected'))} (range {_money_interval(a12)}); expected peak balance "
            f"over the next year {_money(peak.get('expected'))}.{_conf_note(a3, a12)}"
        )
        self.context_blocks = [f"Expected arrears — {name}:\n{line}"]
        self.sources = [self._src(f"Expected arrears — {name}", line)]
        self.artifact = _resident_artifact(self.pred, _HEADS_ARREARS, "arrears")
        self.follow_ups = _follow_ups("arrears")

    def _plan_cure(self) -> None:
        heads = self._heads()
        name = self._name()
        cure = heads.get("p_cure_6m", {})
        mtc = heads.get("months_to_cure", {})
        if cure.get("band") == "not_applicable" or cure.get("probability") is None:
            line = (
                f"Balance cure — {name} currently carries no outstanding balance, so there is "
                f"nothing to cure. The cure estimate applies only to residents currently in arrears."
            )
        else:
            median = mtc.get("median_months")
            median_txt = (
                f"about {median} month(s)" if median else "not within the 12-month window"
            )
            line = (
                f"Balance cure — {name}. Chance the current balance is cleared within 6 months: "
                f"{_pct(cure.get('probability'))} ({cure.get('band')} band, range {_pct_range(cure)}). "
                f"Estimated time to clear: {median_txt}.{_conf_note(cure, mtc)}"
            )
        self.context_blocks = [f"Balance cure — {name}:\n{line}"]
        self.sources = [self._src(f"Balance cure — {name}", line)]
        self.artifact = _resident_artifact(self.pred, _HEADS_CURE, "cure")
        self.follow_ups = _follow_ups("cure")

    def _plan_retention(self) -> None:
        heads = self._heads()
        name = self._name()
        churn = heads.get("churn", {})
        churn12 = heads.get("churn_12m", {})
        mtle = churn.get("months_to_lease_end")
        if churn.get("band") == "not_applicable" and churn12.get("band") == "not_applicable":
            mtle_txt = f"in about {_num(mtle)} months" if mtle else "beyond the renewal horizon"
            line = (
                f"Renewal risk — {name}'s lease ends {mtle_txt}, beyond the renewal-risk horizon, "
                f"so there is no renewal-risk estimate yet. Renewal risk is estimated as the lease "
                f"approaches its end."
            )
        else:
            def band_txt(h, horizon):
                if h.get("band") == "not_applicable" or h.get("probability") is None:
                    return f"{horizon}: not yet in the horizon"
                return f"{horizon}: {_pct(h.get('probability'))} ({h.get('band')} band)"
            line = (
                f"Renewal risk — {name} (chance of NON-renewal). "
                f"{band_txt(churn, 'lease ending within 6 months')}; "
                f"{band_txt(churn12, 'within 12 months')}. Lease ends in about {_num(mtle)} months."
                f"{_conf_note(churn, churn12)} Elevated churn risk is a cue for retention outreach."
            )
        self.context_blocks = [f"Renewal risk — {name}:\n{line}"]
        self.sources = [self._src(f"Renewal risk — {name}", line)]
        self.artifact = _resident_artifact(self.pred, _HEADS_RETENTION, "retention")
        self.follow_ups = _follow_ups("retention")

    # -- portfolio / property planners ---------------------------------------
    def _plan_property_health(self) -> None:
        self.intent = "property_health"
        self.follow_ups = _follow_ups("property_health")
        ranked = portfolio_health()
        self.ranked = ranked
        if self.property_id:
            # This property's health, in the context of the ranking.
            health = next((h for h in ranked if h.get("property_id") == self.property_id), None)
            if health is None:
                health = property_health(self.property_id)
            if health is not None:
                self.health = health
                rank = next((i + 1 for i, h in enumerate(ranked)
                             if h.get("property_id") == self.property_id), None)
                rank_txt = f", ranked {rank} of {len(ranked)} healthiest" if rank else ""
                line = (
                    f"Property health — {health.get('name') or self.property_id} scores "
                    f"{_num(health.get('score'))}/100 (grade {health.get('grade')}) across "
                    f"{health.get('resident_count', 0)} residents{rank_txt}. Biggest drag: "
                    f"{health.get('top_driver') or 'healthy across the board'}."
                )
                self.context_blocks = [f"Property health:\n{line}"]
                self.sources = [{"type": "property_health",
                                 "label": f"Property health — {health.get('name')}",
                                 "snippet": _snippet(line), "property_id": self.property_id}]
                self.artifact = _property_health_artifact(health, ranked)
                return
        # Portfolio ranking (no property scoped, or health lookup failed).
        if not ranked:
            self._plan_general()
            self.intent = "property_health"
            return
        healthiest = ranked[0]
        worst = ranked[-1]
        listing = "; ".join(
            f"{h.get('name') or h.get('property_id')} {_num(h.get('score'))}/100 (grade {h.get('grade')})"
            for h in ranked
        )
        line = (
            f"Portfolio health ranking (healthiest first) across {len(ranked)} properties. "
            f"Healthiest: {healthiest.get('name') or healthiest.get('property_id')} at "
            f"{_num(healthiest.get('score'))}/100 (grade {healthiest.get('grade')}). Needs the most "
            f"attention: {worst.get('name') or worst.get('property_id')} at {_num(worst.get('score'))}/100 "
            f"(grade {worst.get('grade')}), biggest drag {worst.get('top_driver') or 'n/a'}. "
            f"Full ranking: {listing}."
        )
        self.context_blocks = [f"Portfolio health ranking:\n{line}"]
        self.sources = [{"type": "property_health", "label": "Portfolio health ranking",
                         "snippet": _snippet(line), "property_id": ""}]
        self.artifact = _portfolio_health_artifact(ranked)

    def _plan_compare(self, resident_id) -> None:
        self.follow_ups = _follow_ups("compare")
        self.pred = score_resident(resident_id)
        if self.pred is None:
            # Portfolio-scope compare with no resident → fall back to health ranking.
            self._plan_property_health()
            self.intent = "property_health"
            return
        heads = self._heads()
        name = self._name()
        pid = self.pred.get("property_id", "")
        ranked = portfolio_health()
        self.ranked = ranked
        health = next((h for h in ranked if h.get("property_id") == pid), None)
        self.health = health
        late12 = heads.get("late_12m", {})
        serious = heads.get("serious", {})
        res_line = (
            f"Resident vs portfolio — {name}. Their late-payment risk over the next year is "
            f"{_pct(late12.get('probability'))} ({late12.get('band')} band) and serious-delinquency "
            f"risk next quarter is {_pct(serious.get('probability'))} ({serious.get('band')} band)."
            f"{_conf_note(late12, serious)}"
        )
        if health is not None:
            rank = next((i + 1 for i, h in enumerate(ranked)
                         if h.get("property_id") == pid), None)
            rank_txt = f" (ranked {rank} of {len(ranked)} healthiest)" if rank else ""
            prop_line = (
                f"Their property {health.get('name') or pid} scores {_num(health.get('score'))}/100 "
                f"(grade {health.get('grade')}) across {health.get('resident_count', 0)} residents"
                f"{rank_txt}. Portfolio healthiest: {ranked[0].get('name') or ranked[0].get('property_id')} "
                f"at {_num(ranked[0].get('score'))}/100; needs most attention: "
                f"{ranked[-1].get('name') or ranked[-1].get('property_id')} at {_num(ranked[-1].get('score'))}/100."
            )
        else:
            prop_line = "Property-health context is unavailable for this resident's property."
        self.context_blocks = [
            f"Resident vs portfolio — {name}:\n{res_line}",
            f"Property context:\n{prop_line}",
        ]
        self.sources = [
            self._src(f"Resident vs portfolio — {name}", res_line),
            {"type": "property_health", "label": "Property context",
             "snippet": _snippet(prop_line), "property_id": pid},
        ]
        self.artifact = _compare_artifact(self.pred, health, ranked)

    def _plan_governance(self, q: str) -> None:
        self.gov = model_governance(q)
        card = self.gov.get("card") or {}
        excluded = card.get("excluded") or []
        heads = card.get("heads") or []
        card_line = (
            f"Model card — {card.get('name', 'Resident Risk')}. Intended use: "
            f"{card.get('intended_use', '')} It estimates late-payment, frequency, severity, "
            f"arrears, cure, and renewal outcomes from a 5-year rent ledger using {len(heads)} "
            f"gradient-boosted heads, and structurally excludes {len(excluded)} protected-class "
            f"proxy or non-predictive fields. It never uses race, national origin, sex, familial "
            f"status, disability, age, religion, or location."
        )
        self.context_blocks = [f"Model card / feature policy:\n{card_line}"]
        self.sources = [{"type": "model_card",
                         "label": f"Model card — {card.get('name', 'Resident Risk')}",
                         "snippet": _snippet(card_line), "resident_id": ""}]
        lookup = self.gov.get("feature_lookup") or []
        if lookup:
            parts = []
            for item in lookup:
                verb = "IS used" if item.get("status") == "used" else "is NOT used (excluded)"
                parts.append(f"'{item.get('field')}' {verb} — {item.get('reason')}")
            lookup_line = "Feature policy lookup. " + " ".join(parts)
            self.context_blocks.append(f"Feature policy lookup:\n{lookup_line}")
            self.sources.append({"type": "model_card", "label": "Feature policy lookup",
                                 "snippet": _snippet(lookup_line), "resident_id": ""})
        self.artifact = _none_artifact()
        self.follow_ups = _follow_ups("governance")

    def _plan_general(self) -> None:
        self.intent = "general"
        try:
            card = residents_risk.model_card()
            use = card.get("intended_use", "")
        except Exception:  # noqa: BLE001
            use = ""
        line = (
            "Residents decision-support assistant. It explains a resident's late-payment, "
            "frequency, severity, arrears, cure, and renewal estimates and the model's own "
            f"reason codes, and ranks properties by a composite health score. {use}"
        )
        self.context_blocks = [f"About the residents assistant:\n{line}"]
        self.sources = [{"type": "model_card", "label": "About the residents assistant",
                         "snippet": _snippet(line), "resident_id": ""}]
        self.artifact = _none_artifact()
        self.follow_ups = _follow_ups("general")

    # -- deterministic answer (grounded templates; NO approve/deny lexicon) --
    def deterministic_answer(self) -> str:
        try:
            return self._deterministic_answer()
        except Exception as exc:  # noqa: BLE001
            print(f"residents_chat: deterministic answer failed ({type(exc).__name__}: {exc}).")
            return _DEFLECTION

    def _deterministic_answer(self) -> str:
        intent = self.intent
        if intent == "governance" and self.gov is not None:
            return _deterministic_governance(self.gov)
        if intent == "property_health":
            return _deterministic_property_health(self)
        if intent == "general":
            return _GENERAL
        if self.pred is None:
            return _DEFLECTION
        if intent == "compare":
            return _deterministic_compare(self)
        return _deterministic_resident(self)


# ---------------------------------------------------------------------------
# DETERMINISTIC templates (each grounds on the numbered context via [n]).
# ---------------------------------------------------------------------------
def _deterministic_resident(plan: "_ResidentPlan") -> str:
    # The first (and for most intents only) context block already reads as a
    # grounded, cited sentence; wrap it with the outreach guard.
    body = plan.context_blocks[0].split("\n", 1)[-1] if plan.context_blocks else ""
    if plan.intent == "explain" and len(plan.context_blocks) > 1:
        factors = plan.context_blocks[1].split("\n", 1)[-1]
        return f"{body} [1] {factors} [2].{_guard_outreach()}"
    return f"{body} [1].{_guard_outreach()}"


def _deterministic_compare(plan: "_ResidentPlan") -> str:
    res = plan.context_blocks[0].split("\n", 1)[-1] if plan.context_blocks else ""
    prop = plan.context_blocks[1].split("\n", 1)[-1] if len(plan.context_blocks) > 1 else ""
    guard = (
        " This is neutral context for planning outreach — a relative position, not a "
        "ranking that penalizes the resident or the property."
    )
    return f"{res} [1] {prop} [2].{guard}"


def _deterministic_property_health(plan: "_ResidentPlan") -> str:
    body = plan.context_blocks[0].split("\n", 1)[-1] if plan.context_blocks else ""
    if not body:
        return _GENERAL
    guard = (
        " Health scores are decision-support for a regional director planning where to focus "
        "outreach — never a penalty on a property or its residents."
    )
    return f"{body} [1].{guard}"


def _deterministic_governance(gov: dict) -> str:
    lookup = gov.get("feature_lookup") or []
    parts: list[str] = []
    for item in lookup:
        if item.get("status") == "used":
            parts.append(f"'{item.get('field')}' IS used — {item.get('reason')} [1]")
        else:
            parts.append(f"'{item.get('field')}' is NOT used — {item.get('reason')} [1]")
    card = gov.get("card") or {}
    heads = card.get("heads") or []
    excluded = card.get("excluded") or []
    base = (
        f"The model estimates outcomes from a 5-year rent ledger using {len(heads)} "
        f"gradient-boosted heads (late-payment, frequency, severity, arrears, cure, and "
        f"renewal). It uses only legitimate ledger, tenancy, and affordability factors and "
        f"structurally excludes {len(excluded)} protected-class proxy or non-predictive fields; "
        f"it never uses race, national origin, sex, familial status, disability, age, religion, "
        f"or location [1]. Reason codes come from the model's own TreeSHAP attributions. This is "
        f"decision-support on synthetic data — an estimate, not a consumer report."
    )
    if parts:
        return " ".join(parts) + ". " + base
    return base


# ---------------------------------------------------------------------------
# LLM synthesis (optional; None offline or on any error) — mirrors risk_chat.
# ---------------------------------------------------------------------------
def _coalesce(content) -> str:
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
    numbered = "\n\n".join(f"[{i+1}] {block}" for i, block in enumerate(context_blocks))
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


def _llm_answer(question, context_blocks, history, system=_RESIDENT_SYSTEM) -> str | None:
    llm = get_langchain_llm()
    if llm is None:
        return None
    try:
        messages = _build_messages(question, context_blocks, history, system)
        raw = llm.invoke(messages).content
        text = _coalesce(raw).strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 — degrade to deterministic
        print(f"residents_chat: LLM synthesis failed ({type(exc).__name__}); templated.")
        return None


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT — never raises
# ---------------------------------------------------------------------------
def answer(question: str, resident_id: str | None = None,
           property_id: str | None = None, history=None) -> dict:
    try:
        return _answer(question, resident_id, property_id, history)
    except Exception as exc:  # noqa: BLE001 — mirror risk_chat.answer: never 500
        print(f"residents_chat: unexpected error ({type(exc).__name__}: {exc}); safe fallback.")
        try:
            intent = route(question or "", resident_id, property_id)
        except Exception:  # noqa: BLE001
            intent = "general"
        return {
            "answer": (
                "Sorry, I hit a snag looking that up. Please try rephrasing, or select a "
                "resident to see their estimates, or ask which properties are healthiest."
            ),
            "scope": "resident" if resident_id else ("property" if property_id else "portfolio"),
            "resident_id": resident_id or "",
            "property_id": property_id or "",
            "intent": intent,
            "sources": [],
            "artifact": {"kind": "none"},
            "follow_ups": _follow_ups(intent),
            "source": "rules",
        }


def _answer(question: str, resident_id, property_id, history) -> dict:
    plan = _ResidentPlan(question, resident_id, property_id)

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
        "resident_id": resident_id or "",
        "property_id": property_id or "",
        "intent": plan.intent,
        "sources": plan.sources,
        "artifact": plan.artifact,
        "follow_ups": plan.follow_ups,
        "source": source,
    }


# ---------------------------------------------------------------------------
# STREAMING ENTRY POINT — a generator that can never crash the response
# ---------------------------------------------------------------------------
def answer_stream(question: str, resident_id: str | None = None,
                  property_id: str | None = None, history=None):
    """Yield SSE event dicts: one ``meta`` (the deterministic ``_ResidentPlan``
    pass, sent up front so the UI can paint gauges/lists), then ``token`` events
    streaming the LLM prose, then a final ``done``. Reuses ``_ResidentPlan`` so
    the streamed artifact/meta are IDENTICAL to ``answer()``. On any failure it
    degrades to a single deterministic ``token`` + ``done`` (source="rules").
    NEVER raises."""
    scope = "resident" if resident_id else ("property" if property_id else "portfolio")
    try:
        plan = _ResidentPlan(question, resident_id, property_id)
    except Exception as exc:  # noqa: BLE001 — never crash the stream
        print(f"residents_chat: stream planning failed ({type(exc).__name__}: {exc}).")
        try:
            intent = route(question or "", resident_id, property_id)
        except Exception:  # noqa: BLE001
            intent = "general"
        yield {
            "type": "meta", "scope": scope, "resident_id": resident_id or "",
            "property_id": property_id or "", "intent": intent,
            "follow_ups": _follow_ups(intent), "artifact": {"kind": "none"},
        }
        yield {
            "type": "token",
            "text": (
                "Sorry, I hit a snag looking that up. Please try rephrasing, or select a "
                "resident to see their estimates."
            ),
        }
        yield {"type": "done", "source": "rules"}
        return

    # META first — the exact field set the frontend expects (same artifact dict
    # answer() returns).
    yield {
        "type": "meta", "scope": plan.scope, "resident_id": resident_id or "",
        "property_id": property_id or "", "intent": plan.intent,
        "follow_ups": plan.follow_ups, "artifact": plan.artifact,
    }

    streamed_any = False
    if plan.context_blocks:
        llm = get_langchain_llm()
        if llm is not None:
            try:
                messages = _build_messages(question, plan.context_blocks, history, plan.system)
                for chunk in llm.stream(messages):
                    text = _coalesce(getattr(chunk, "content", ""))
                    if text:
                        streamed_any = True
                        yield {"type": "token", "text": text}
                if streamed_any:
                    yield {"type": "done", "source": "anthropic"}
                    return
            except Exception as exc:  # noqa: BLE001 — degrade to deterministic
                print(f"residents_chat: stream synthesis failed ({type(exc).__name__}).")

    try:
        det = plan.deterministic_answer()
    except Exception as exc:  # noqa: BLE001
        print(f"residents_chat: deterministic fallback failed ({type(exc).__name__}).")
        det = (
            "Sorry, I hit a snag looking that up. Please try rephrasing, or select a "
            "resident to see their estimates."
        )
    yield {"type": "token", "text": det}
    yield {"type": "done", "source": "rules"}
