"""Pydantic schemas shared across the app.

Defining these as typed models means the LLM extraction, the eligibility
rules, the scoring, and the API responses all speak the same validated
language.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


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
        "greeting", "proposing", "confirming", "awaiting_name", "booked", "no_availability"
    ] = "greeting"
    prospect_name: str = ""
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
