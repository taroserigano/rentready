"""Pydantic schemas shared across the app.

Defining these as typed models means the LLM extraction, the eligibility
rules, the scoring, and the API responses all speak the same validated
language.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ApplicantProfile(BaseModel):
    """Everything we pull out of an uploaded application PDF.

    Hard constraints: has_pets (must allow), affordability (income/rent).
    Everything else is a soft preference that influences the match score.
    """

    name: str = Field(default="Unknown")
    monthly_income: float = Field(default=0.0, description="Gross monthly income")
    desired_rent: float = Field(default=0.0, description="Rent the applicant wants")
    credit_score: Optional[int] = Field(default=None)
    employment_status: str = Field(default="unknown")

    # Household / unit preferences
    bedrooms_wanted: Optional[int] = Field(default=None)
    bathrooms_wanted: Optional[float] = Field(default=None)
    bath_type_wanted: Literal["full", "shower_only", "any"] = Field(default="any")
    min_square_feet: Optional[int] = Field(default=None)

    # Lifestyle preferences
    has_pets: bool = Field(default=False)
    needs_balcony: bool = Field(default=False)
    needs_parking: bool = Field(default=False)
    needs_in_unit_laundry: bool = Field(default=False)
    furnished_wanted: bool = Field(default=False)

    # Location preferences
    preferred_area: str = Field(default="")
    wanted_amenities: list[str] = Field(default_factory=list)
    lease_term_wanted: Optional[int] = Field(default=None, description="Months")

    # Employment & income detail
    employer: str = Field(default="", description="Current employer name")
    job_title: str = Field(default="", description="Current job title")
    employment_length_months: Optional[int] = Field(
        default=None, ge=0, description="Months at current job"
    )
    other_income_monthly: float = Field(
        default=0, ge=0, description="Monthly income outside the main job"
    )
    savings_balance: Optional[float] = Field(
        default=None, ge=0, description="Total savings on hand, in dollars"
    )
    monthly_debt_payments: float = Field(
        default=0, ge=0, description="Total monthly debt payments (loans, cards)"
    )

    # Housing history
    current_address: str = Field(default="", description="Where the applicant lives now")
    current_rent: Optional[float] = Field(
        default=None, ge=0, description="Rent paid at the current home"
    )
    years_at_current_address: Optional[float] = Field(
        default=None, ge=0, description="Years lived at the current address"
    )
    reason_for_moving: str = Field(default="", description="Why the applicant is moving")
    landlord_reference: bool = Field(
        default=False, description="Current landlord will give a reference"
    )

    # Screening history
    evictions_count: int = Field(default=0, ge=0, description="Number of past evictions")
    late_payments_12mo: int = Field(
        default=0, ge=0, description="Late rent payments in the last 12 months"
    )
    bankruptcies_count: int = Field(
        default=0, ge=0, description="Number of past bankruptcies"
    )
    criminal_record: bool = Field(default=False, description="Has a criminal record")
    smoker: bool = Field(default=False, description="Applicant smokes")

    # Household composition
    household_size: int = Field(
        default=1, ge=1, description="Total people who will live in the unit"
    )
    co_applicants: int = Field(
        default=0, ge=0, description="Other adults applying on the same lease"
    )
    dependents: int = Field(default=0, ge=0, description="Children or other dependents")
    pet_count: int = Field(default=0, ge=0, description="Number of pets")
    pet_types: list[str] = Field(
        default_factory=list, description="Kinds of pets, e.g. dog, cat"
    )
    vehicles_count: int = Field(default=0, ge=0, description="Vehicles needing parking")

    # Application logistics
    desired_move_in: str = Field(
        default="", description="Move-in date the applicant wants (ISO date)"
    )
    guarantor_available: bool = Field(
        default=False, description="Someone can co-sign / guarantee the lease"
    )
    references_count: int = Field(
        default=0, ge=0, description="Personal or professional references offered"
    )
    is_student: bool = Field(default=False, description="Applicant is a student")


class EligibilityResult(BaseModel):
    verdict: Literal["qualified", "needs_review", "not_qualified"]
    reasons: list[str]
    income_to_rent_ratio: Optional[float] = None
    explanation: str = ""


class PropertyRecommendation(BaseModel):
    property_id: str
    name: str
    area: str
    property_type: str = ""
    monthly_rent: float
    bedrooms: int
    bathrooms: float = 0.0
    bathroom_type: str = ""
    square_feet: int = 0
    has_balcony: bool = False
    in_unit_laundry: bool = False
    pets_allowed: bool = False
    parking_type: str = ""
    walk_score: Optional[int] = None
    transit_score: Optional[int] = None
    amenities: list[str] = Field(default_factory=list)
    photo_url: Optional[str] = None
    photo_urls: list[str] = Field(default_factory=list)
    match_reason: str = ""
    fit_highlights: list[str] = Field(default_factory=list)
    score: float = 0.0
    signal_breakdown: dict = Field(default_factory=dict)


class UploadResponse(BaseModel):
    applicant_id: str
    profile: ApplicantProfile
    chunks_indexed: int
    # Whether an application PDF is on file (served at /applicants/{id}/pdf).
    has_pdf: bool = False


class AskRequest(BaseModel):
    applicant_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    source: str
    sources: list[str] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    recommendations: list[PropertyRecommendation]
    source: str
    graph_backend: str
    relaxed: bool = False
    # Signal weights used by the deterministic scorer, so the UI can show the
    # exact score math ("Why this rank?"). Same for every response.
    weights: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tour Scheduler (F: book a property tour with a leasing agent)
# ---------------------------------------------------------------------------


class TourAgent(BaseModel):
    """A leasing agent who runs tours."""

    id: str
    name: str
    role: str = "Leasing Consultant"
    # Neighborhood names this agent covers; [] means "covers all areas".
    areas: list[str] = Field(default_factory=list)


class AvailabilityWindow(BaseModel):
    """A recurring weekly window an agent is available for tours.

    ``weekday`` is 0=Monday .. 6=Sunday. ``start``/``end`` are local "HH:MM".
    """

    agent_id: str
    weekday: int
    start: str  # "HH:MM"
    end: str  # "HH:MM"


class Slot(BaseModel):
    """A single bookable 30-minute tour slot, tied to one agent."""

    slot_id: str  # STABLE id = f"{agent_id}|{start_iso}"
    property_id: str
    start: str  # ISO local "YYYY-MM-DDTHH:MM:SS"
    end: str  # ISO local
    agent_id: str
    agent_name: str
    label: str  # human, e.g. "Tue Jul 22, 2:00 PM"


class TourBooking(BaseModel):
    id: str  # "TOUR-xxxxxxxxxxxx"
    property_id: str
    property_name: str  # denormalized for UI
    start: str  # ISO local
    end: str
    duration_minutes: int = 30
    agent_id: str
    agent_name: str  # denormalized
    prospect_name: str
    prospect_email: str = ""  # "" if unknown
    prospect_phone: str = ""  # "" if unknown
    status: Literal["booked", "cancelled"] = "booked"
    created_at: str  # UTC ISO
    # "Add to Google Calendar" template link (self-contained; no server creds).
    # Populated by the API layer; empty on rows straight from the store.
    gcal_url: str = ""


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatState(BaseModel):
    phase: Literal[
        "greeting", "proposing", "confirming", "awaiting_name",
        "awaiting_phone", "awaiting_email", "booked", "no_availability",
        "awaiting_cancel_email",
    ] = "greeting"
    prospect_name: str = ""
    prospect_phone: str = ""
    prospect_email: str = ""
    last_proposed: list[Slot] = Field(default_factory=list)
    pending_slot_id: str = ""


class TourChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    property_id: Optional[str] = None
    state: Optional[ChatState] = None
    selected_slot_id: Optional[str] = None


class TourChatResponse(BaseModel):
    reply: str
    proposed_slots: list[Slot] = Field(default_factory=list)
    booking: Optional[TourBooking] = None
    state: ChatState = Field(default_factory=ChatState)
    source: Literal["rules", "anthropic"] = "rules"


class BookTourRequest(BaseModel):
    property_id: str
    slot_id: str
    prospect_name: str = ""
    prospect_email: str = ""
    prospect_phone: str = ""
    applicant_id: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Resident Late-Payment Risk (decision-support; synthetic-data model)
# ---------------------------------------------------------------------------


class ReasonCode(BaseModel):
    """One key factor behind a risk estimate (FCRA-style). ``contribution`` is
    the signed SHAP / heuristic value; render |contribution| for bar length."""

    feature: str
    label: str
    direction: Literal["increases", "decreases"]
    contribution: float


class RiskResult(BaseModel):
    applicant_id: str = ""
    name: Optional[str] = None
    probability: float  # calibrated P(late), 0..1
    band: Literal["low", "medium", "high"]
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    confidence: Literal["high", "low"]
    range: list[float] = Field(default_factory=list)  # [lo, hi]
    source: Literal["model", "heuristic"]
    model_type: Literal["xgboost", "histgb", "heuristic"]
    scored_at: Optional[str] = None


class RiskScoreRequest(BaseModel):
    """Score an ad-hoc profile (e.g. a what-if) without persisting it."""

    profile: ApplicantProfile


# ---------------------------------------------------------------------------
# Residents — current-resident risk (5-year rent ledger; multi-target model)
#
# Committed records store ONLY the rent ledger + immutable facts. Everything
# time-relative (current balance, tenure, arrears, response rate) is DERIVED at
# scoring time from the ledger + a pinned snapshot date, so training and serving
# see the exact same computation and no future information leaks in.
# ---------------------------------------------------------------------------


class LedgerEntry(BaseModel):
    """One month of a resident's rent ledger. Oldest→newest, ending at (and
    never after) the snapshot month. This is a committed fact — payment
    outcomes only, no derived/forward-looking fields."""

    period: str  # "YYYY-MM"
    rent_charged: float = Field(ge=0.0)
    amount_paid: float = Field(ge=0.0)
    paid_date: Optional[str] = None  # ISO date; None when nothing was paid
    days_late: int = Field(default=0, ge=0)
    late_fee: float = Field(default=0.0, ge=0.0)
    on_time: bool = True
    status: Literal["paid", "paid_late", "partial", "missed"] = "paid"
    balance_after: float = 0.0  # running arrears after this month (>=0)
    notices_sent: int = Field(default=0, ge=0)
    notice_responded: int = Field(default=0, ge=0)


class Resident(BaseModel):
    """A current resident: committed ledger + immutable facts ONLY.

    Time-relative statistics (current balance, tenure, arrears, notice-response
    rate) are intentionally NOT stored — they are derived at scoring time from
    ``ledger`` + the snapshot so train/serve cannot skew and the future cannot
    leak. Protected-class attributes are never stored here at all."""

    # Identity / unit
    resident_id: str  # "RES-0001"
    name: str = ""  # synthetic display name — DISPLAY ONLY, never a model feature
    property_id: str
    unit_id: str
    unit_bedrooms: int = Field(default=1, ge=0)
    base_rent: float = Field(ge=0.0)

    # Lease
    lease_start: str  # ISO date
    lease_end: str  # ISO date
    lease_term_months: int = Field(default=12, ge=1)
    renewal_offer_sent: bool = False
    autopay_enrolled: bool = False
    deposit_held: float = Field(default=0.0, ge=0.0)
    move_in_date: str  # ISO date
    prior_renewals: int = Field(default=0, ge=0)

    # Financial (snapshot facts)
    monthly_income: float = Field(default=0.0, ge=0.0)
    other_income_monthly: float = Field(default=0.0, ge=0.0)
    income_verified: bool = False

    # Engagement (stored counts, immutable facts as of snapshot)
    maintenance_requests_12mo: int = Field(default=0, ge=0)
    complaints_12mo: int = Field(default=0, ge=0)
    portal_logins_90d: int = Field(default=0, ge=0)

    # History
    ledger: list[LedgerEntry] = Field(default_factory=list)

    dgp_version: str = ""


# ---------------------------------------------------------------------------
# Risk Chat — the decision-support agent embedded in the Risk page.
# Mirrors the concierge request/response shapes. The ``artifact`` is a plain
# dict built by ``risk_agent``; its runtime shape is a discriminated union
# keyed on ``kind`` (none | score | reasons | whatif | counterfactual |
# comparison) — see RISK_CHAT_CONTRACT.md. It is intentionally NOT over-modeled
# here so the backend can add union members without a schema migration.
# ---------------------------------------------------------------------------
class RiskChatMessage(BaseModel):
    """One turn of prior chat history."""

    role: Literal["user", "assistant"]
    content: str


class RiskChatRequest(BaseModel):
    """A question for the risk chat agent. Omit ``applicant_id`` for portfolio
    scope. Unknown ids do NOT 404 — the agent deflects gracefully."""

    question: str
    applicant_id: Optional[str] = None
    history: Optional[list[RiskChatMessage]] = None


class RiskChatResponse(BaseModel):
    """The risk chat agent's answer. ALWAYS returned with a 200 — the agent
    never raises. ``source`` is ``"anthropic"`` when the LLM synthesized the
    prose, else ``"rules"`` (deterministic / offline)."""

    answer: str
    scope: str
    applicant_id: str = ""
    intent: str
    sources: list[dict] = Field(default_factory=list)
    artifact: dict = Field(default_factory=lambda: {"kind": "none"})
    follow_ups: list[str] = Field(default_factory=list)
    source: Literal["rules", "anthropic"] = "rules"


# ---------------------------------------------------------------------------
# Residents API — response schemas.
#
# These mirror ``residents_risk.predict_resident``'s output faithfully (one
# sub-model per target) plus the portfolio/rollup aggregates the API layer
# computes. Endpoints ALWAYS return 200 (the model degrades to a transparent
# heuristic server-side); the only typed error is 404 for a genuinely unknown
# resident id on the detail route.
# ---------------------------------------------------------------------------


class LatePrediction(BaseModel):
    """P(any late payment next quarter) — calibrated classifier."""

    probability: float
    band: Literal["low", "medium", "high"]
    range: list[float] = Field(default_factory=list)  # [lo, hi]
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    confidence: Literal["high", "low"]
    source: Literal["model", "heuristic"]
    model_type: str = "heuristic"


class ArrearsPrediction(BaseModel):
    """Expected $ balance at the end of next quarter — regressor + interval."""

    expected_balance: float
    interval: list[float] = Field(default_factory=list)  # [lo, hi]
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    confidence: Literal["high", "low"]
    source: Literal["model", "heuristic"]
    model_type: str = "heuristic"


class ChurnPrediction(BaseModel):
    """P(non-renewal) for leases ending within the horizon; ``probability`` is
    None and ``band`` is ``"not_applicable"`` when the lease ends further out."""

    probability: Optional[float] = None  # None when not applicable
    band: Literal["low", "medium", "high", "not_applicable"]
    months_to_lease_end: Optional[float] = None
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    confidence: Literal["high", "low"]
    source: Literal["model", "heuristic", "not_applicable"]


class SeriousPrediction(BaseModel):
    """P(serious delinquency: 30+ days late or a full month in arrears). Always
    routes to a human reviewer — never an automated action."""

    probability: float
    band: Literal["low", "medium", "high"]
    range: list[float] = Field(default_factory=list)
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    confidence: Literal["high", "low"]
    source: Literal["model", "heuristic"]
    model_type: str = "heuristic"
    routes_to_review: bool = True


class ResidentPredictions(BaseModel):
    """The full multi-HEAD result for one resident — mirrors the dict returned by
    ``residents_risk.predict_resident``. The four legacy sub-models
    (late/arrears/churn/serious) are the backward-compatible aliases; the full v2
    catalog rides along in ``heads`` (per-head payloads, kind-specific) grouped by
    ``families``. ``heads`` is a loose dict passthrough so new heads never force a
    schema migration."""

    resident_id: str = ""
    property_id: str = ""
    name: str = ""  # synthetic display name (DISPLAY ONLY)
    snapshot_date: str = ""
    late: LatePrediction
    arrears: ArrearsPrediction
    churn: ChurnPrediction
    serious: SeriousPrediction
    heads: dict = Field(default_factory=dict)  # {head_name: kind-specific payload}
    families: dict = Field(default_factory=dict)  # {family: [head_name, ...]}
    scored_at: Optional[str] = None


class ResidentRow(BaseModel):
    """One row of the portfolio residents table — a flattened, at-a-glance view
    of the four predictions for a single resident."""

    resident_id: str
    property_id: str
    unit_id: str
    name: str = ""
    base_rent: float
    tenure_months: int
    late_probability: float
    late_band: Literal["low", "medium", "high"]
    expected_arrears: float
    churn_probability: Optional[float] = None
    churn_status: Literal["low", "medium", "high", "not_applicable"]
    serious_probability: float
    serious_band: Literal["low", "medium", "high"]
    current_balance: float
    top_driver: str = ""


class ResidentListResponse(BaseModel):
    residents: list[ResidentRow] = Field(default_factory=list)
    count: int = 0
    property_id: Optional[str] = None  # set when the list was filtered
    source: Literal["model", "heuristic"] = "heuristic"


class LedgerStats(BaseModel):
    """Time-relative statistics DERIVED from the ledger + snapshot at read time
    (never stored on the record). Computed via the same feature extraction the
    models use, so what the UI shows and what the model saw cannot diverge."""

    ledger_months: int
    tenure_months: int
    current_balance: float
    current_balance_ratio: float
    balance_trend_6mo: float
    times_late_3mo: int
    times_late_6mo: int
    times_late_12mo: int
    times_late_24mo: int
    missed_count_12mo: int
    partial_count_12mo: int
    on_time_streak_months: int
    months_since_last_late: int
    max_days_late_12mo: int
    avg_days_late_12mo: float
    late_fees_12mo: float
    notice_response_rate: float
    rent_to_income: float
    autopay_enrolled: bool
    income_verified: bool
    lifetime_paid: float
    lifetime_late_fees: float


class ResidentDetail(BaseModel):
    """Full drill-down for one resident: the committed record, the four
    predictions, and derived ledger statistics."""

    resident: Resident
    predictions: ResidentPredictions
    ledger_stats: LedgerStats
    source: Literal["model", "heuristic"] = "heuristic"


class BandDistribution(BaseModel):
    low: int = 0
    medium: int = 0
    high: int = 0
    not_applicable: int = 0  # churn only


class ResidentRollup(BaseModel):
    """Aggregate view over a set of residents (one property, or the portfolio)."""

    resident_count: int = 0
    predicted_late_rate: float = 0.0  # mean P(late) next quarter
    total_expected_arrears: float = 0.0  # sum of expected balances next quarter
    avg_serious_probability: float = 0.0
    churn_eligible_count: int = 0  # leases ending within the churn horizon
    churn_risk_count: int = 0  # of eligible, band == "high"
    serious_flag_count: int = 0  # band == "high" -> routes to review
    late_bands: BandDistribution = Field(default_factory=BandDistribution)
    serious_bands: BandDistribution = Field(default_factory=BandDistribution)
    churn_bands: BandDistribution = Field(default_factory=BandDistribution)


class PropertyResidentRollup(ResidentRollup):
    property_id: str


class PropertyResidentsResponse(BaseModel):
    property_id: str
    residents: list[ResidentRow] = Field(default_factory=list)
    count: int = 0
    rollup: PropertyResidentRollup
    source: Literal["model", "heuristic"] = "heuristic"


class ResidentPropertyOption(BaseModel):
    """One entry in the property selector — cheap (no scoring): id, display name,
    and how many residents live there. Powers the Residents-page picker so the
    UI never loads/scores the whole portfolio just to render the selector."""

    property_id: str
    name: str
    resident_count: int


class ResidentPropertiesResponse(BaseModel):
    properties: list[ResidentPropertyOption] = Field(default_factory=list)
    count: int = 0


class PortfolioSummary(BaseModel):
    properties: list[PropertyResidentRollup] = Field(default_factory=list)
    overall: ResidentRollup = Field(default_factory=ResidentRollup)
    property_count: int = 0
    resident_count: int = 0
    snapshot_date: str = ""
    source: Literal["model", "heuristic"] = "heuristic"
    generated_at: Optional[str] = None


class ResidentModelCard(BaseModel):
    """The resident-risk model card. Loosely typed (``extra='allow'``) so the
    rich per-head detail from ``residents_risk.model_card`` passes through
    untouched for the frontend."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    version: str = ""
    description: str = ""
    intended_use: str = ""
    targets: list[dict] = Field(default_factory=list)
    heads: list[dict] = Field(default_factory=list)
    families: dict = Field(default_factory=dict)
    excluded: list[dict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source: str = "heuristic"


# ---------------------------------------------------------------------------
# Property / portfolio health (regional-director best->worst ranking).
# Mirrors ``residents_risk.property_health`` / ``portfolio_health``. Loosely
# typed so the components/drivers detail passes through for the frontend.
# ---------------------------------------------------------------------------
class PropertyHealth(BaseModel):
    """Composite 0-100 health score (+ letter grade A-F) for one property, from
    its residents' predictions. Higher = healthier."""

    model_config = ConfigDict(extra="allow")

    property_id: str
    name: str = ""  # property display name (filled by the API layer)
    score: float = 0.0
    grade: Literal["A", "B", "C", "D", "F"] = "F"
    resident_count: int = 0
    top_driver: str = ""
    drivers: list[dict] = Field(default_factory=list)
    components: dict = Field(default_factory=dict)


class PortfolioHealthResponse(BaseModel):
    """Ranked property-health list (healthiest first) for the regional director,
    with an explicit worst-property callout."""

    properties: list[PropertyHealth] = Field(default_factory=list)
    count: int = 0
    healthiest: Optional[PropertyHealth] = None
    needs_attention: Optional[PropertyHealth] = None  # worst-scoring property
    snapshot_date: str = ""
    source: Literal["model", "heuristic"] = "heuristic"


# ---------------------------------------------------------------------------
# Residents Chat — the decision-support agent embedded in the Residents page.
# Mirrors ``RiskChat*`` (and the concierge shapes). The ``artifact`` is a plain
# dict built by ``residents_chat``; its runtime shape is a discriminated union
# keyed on ``kind`` (none | resident | property_health | compare) carrying the
# RELEVANT head payloads (a resident's predictions, or a property-health
# ranking) so the UI can render gauges / lists inline. Intentionally NOT
# over-modeled so the backend can add union members without a schema migration.
# ---------------------------------------------------------------------------
class ResidentChatMessage(BaseModel):
    """One turn of prior chat history."""

    role: Literal["user", "assistant"]
    content: str


class ResidentChatRequest(BaseModel):
    """A question for the residents chat agent. ``resident_id`` scopes to one
    resident; ``property_id`` (or no id) scopes to a property / portfolio health
    view. Unknown ids do NOT 404 — the agent deflects gracefully."""

    question: str
    resident_id: Optional[str] = None
    property_id: Optional[str] = None
    history: Optional[list[ResidentChatMessage]] = None


class ResidentChatResponse(BaseModel):
    """The residents chat agent's answer. ALWAYS returned with a 200 — the agent
    never raises. ``source`` is ``"anthropic"`` when the LLM synthesized the
    prose, else ``"rules"`` (deterministic / offline). Every number in ``answer``
    originates in ``artifact`` (the head payloads); the LLM never computes one."""

    answer: str
    scope: str
    resident_id: str = ""
    property_id: str = ""
    intent: str
    sources: list[dict] = Field(default_factory=list)
    artifact: dict = Field(default_factory=lambda: {"kind": "none"})
    follow_ups: list[str] = Field(default_factory=list)
    source: Literal["rules", "anthropic"] = "rules"
