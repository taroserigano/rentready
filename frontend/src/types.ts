export interface ApplicantProfile {
  name: string;
  monthly_income: number;
  desired_rent: number;
  credit_score: number | null;
  employment_status: string;
  bedrooms_wanted: number | null;
  bathrooms_wanted: number | null;
  bath_type_wanted: string;
  min_square_feet: number | null;
  has_pets: boolean;
  needs_balcony: boolean;
  needs_parking: boolean;
  needs_in_unit_laundry: boolean;
  furnished_wanted: boolean;
  preferred_area: string;
  wanted_amenities: string[];
  lease_term_wanted: number | null;
  // Employment (optional; backend defaults shown in comments)
  employer?: string; // ""
  job_title?: string; // ""
  employment_length_months?: number | null;
  other_income_monthly?: number; // 0
  // Financial
  savings_balance?: number | null;
  monthly_debt_payments?: number; // 0
  // Rental history
  current_address?: string; // ""
  current_rent?: number | null;
  years_at_current_address?: number | null;
  reason_for_moving?: string; // ""
  landlord_reference?: boolean; // false
  evictions_count?: number; // 0
  late_payments_12mo?: number; // 0
  bankruptcies_count?: number; // 0
  criminal_record?: boolean; // false
  // Household
  smoker?: boolean; // false
  household_size?: number; // 1
  co_applicants?: number; // 0
  dependents?: number; // 0
  pet_count?: number; // 0
  pet_types?: string[]; // []
  vehicles_count?: number; // 0
  is_student?: boolean; // false
  // Move-in
  desired_move_in?: string; // ""
  guarantor_available?: boolean; // false
  references_count?: number; // 0
}

export interface UploadResponse {
  applicant_id: string;
  profile: ApplicantProfile;
  chunks_indexed: number;
  has_pdf?: boolean;
}

export interface EligibilityResult {
  verdict: "qualified" | "needs_review" | "not_qualified";
  reasons: string[];
  income_to_rent_ratio: number | null;
  explanation: string;
}

/**
 * One rentable layout inside a property (e.g. "Studio", "2 Bedroom").
 * Houses/townhouses/duplexes have exactly one plan; apartment-style
 * properties can have several. The property's top-level bedrooms /
 * bathrooms / square_feet / monthly_rent always mirror the CHEAPEST plan.
 */
export interface FloorPlan {
  name: string;
  /** 0 means studio. */
  bedrooms: number;
  bathrooms: number;
  square_feet: number;
  monthly_rent: number;
  /** How many units of this plan are open right now (0-8). */
  available_units: number;
  /** ISO date, e.g. "2026-08-01". */
  availability_date: string;
}

/**
 * New listing fields shared by Property and PropertyRecommendation.
 * All optional: older cached rows may not carry them, so the UI must
 * render nothing (never "undefined") when a value is missing.
 */
export interface PropertyExtras {
  /** Listing photo (Unsplash CDN URL). */
  photo_url?: string;
  /** Full listing gallery (Unsplash CDN URLs); first is usually the hero. */
  photo_urls?: string[];
  floor_plans?: FloorPlan[];
  /** Convenience counts the backend may add alongside floor_plans. */
  plan_count?: number;
  max_bedrooms?: number;
  year_built?: number;
  security_deposit?: number;
  application_fee?: number;
  admin_fee?: number;
  pet_deposit?: number;
  pet_rent_monthly?: number;
  pet_weight_limit_lbs?: number | null;
  pets_max_count?: number;
  parking_fee_monthly?: number;
  utilities_included?: string[];
  appliances?: string[];
  flooring?: string;
  heating?: string;
  cooling?: string;
  laundry_type?: string;
  floor_level?: number | null;
  unit_count?: number;
  availability_date?: string;
  min_credit_score?: number;
  min_income_multiplier?: number;
  smoking_allowed?: boolean;
  max_occupants?: number;
  storage_unit_available?: boolean;
  gated?: boolean;
}

export interface PropertyRecommendation extends PropertyExtras {
  property_id: string;
  name: string;
  area: string;
  property_type: string;
  monthly_rent: number;
  bedrooms: number;
  bathrooms: number;
  bathroom_type: string;
  square_feet: number;
  has_balcony: boolean;
  in_unit_laundry: boolean;
  pets_allowed: boolean;
  parking_type: string;
  walk_score: number | null;
  transit_score: number | null;
  amenities: string[];
  match_reason: string;
  fit_highlights: string[];
  score: number;
  signal_breakdown: Record<string, number>;
}

export interface RecommendResponse {
  recommendations: PropertyRecommendation[];
  source: string;
  graph_backend: string;
  relaxed: boolean;
  /** Deterministic scorer weights per signal (for the "Why this rank?" math). */
  weights?: Record<string, number>;
}

export interface SimulateResponse {
  eligibility: EligibilityResult;
  recommendations: RecommendResponse;
}

export interface AskResponse {
  answer: string;
  source: string;
  sources: string[];
}

// --- Property & Lease Concierge --------------------------------------------

export interface Source {
  type: "property" | "lease";
  label: string;
  snippet: string;
  /** Lease section title this source came from; "" for property sources. */
  section?: string;
  /** Property this source belongs to (enables deep-linking to the lease). */
  property_id?: string;
}

/**
 * One matched home in a cross-property comparison. `matched` holds the
 * human-readable reasons the filter fired, e.g. ["under $2,000","allows pets"].
 */
export interface CompareItem {
  id: string;
  name: string;
  area: string;
  property_type: string;
  monthly_rent: number;
  bedrooms: number;
  bathrooms: number;
  square_feet: number;
  pets_allowed: boolean;
  matched: string[];
}

export interface ConciergeAnswer {
  answer: string;
  route: string;
  sources: Source[];
  source: string;
  property_id: string;
  /** 0–3 suggested next questions. */
  follow_ups: string[];
  /** Cross-property comparison matches; [] when the answer isn't a comparison. */
  comparison: CompareItem[];
}

/** First SSE frame of a streamed answer — the deterministic retrieval pass. */
export interface ConciergeMeta {
  route: string;
  property_id: string;
  sources: Source[];
  follow_ups: string[];
  comparison: CompareItem[];
}

/** One row in the Concierge eval per-item breakdown. */
export interface ConciergeEvalItem {
  id: string;
  question: string;
  expected_route: string;
  predicted_route: string;
  route_ok: boolean;
  retrieval_ok: boolean;
  grounded_ok: boolean;
}

/** Result of POST /evals/concierge/run — three metrics in [0,1] + per-item rows. */
export interface ConciergeEvalResult {
  route_accuracy: number;
  retrieval_hit_rate: number;
  groundedness: number;
  n: number;
  generated_at: string;
  items: ConciergeEvalItem[];
}

/** Structured lease facts summarized as a key-terms row in the viewer. */
export interface LeaseKeyTerms {
  rent: number;
  deposit: number;
  term_months: number;
  pets: boolean;
  parking: string;
  furnished: boolean;
}

/** Full lease document for a property (from GET /concierge/lease/{id}). */
export interface LeaseDoc {
  property_id: string;
  property_name: string;
  sections: { section: string; text: string }[];
  key_terms: LeaseKeyTerms;
}

export interface ApplicantSummary {
  id: string;
  name: string;
  chunks_indexed: number;
  created_at: string;
  status?: string;
}

export interface ScreenCheck {
  label: string;
  required: string;
  actual: string;
  ok: boolean | null;
}

export interface ScreenResult {
  property_id: string;
  applicant_id: string;
  passes: boolean;
  verdict: string;
  checks: ScreenCheck[];
}

export interface Candidate {
  applicant_id: string;
  name: string;
  score: number;
  signal_breakdown: Record<string, number>;
  screen_passes: boolean;
  screen_verdict: string;
}

export interface CandidatesResponse {
  property_id: string;
  name: string;
  candidates: Candidate[];
  total: number;
}

export interface StrengthFactor {
  label: string;
  points: number;
  max: number;
  suggestion: string;
}

export interface StrengthResult {
  score: number;
  band: string;
  factors: StrengthFactor[];
  suggestions: string[];
}

export interface GoalSeekResult {
  solve_for: string;
  current: number;
  threshold?: number;
  achievable: boolean;
  delta?: number;
  reason?: string;
}

export interface DecisionItem {
  ts: string;
  action: string;
  note: string | null;
  reviewer: string | null;
}

export interface Property extends PropertyExtras {
  id: string;
  name: string;
  property_type: string;
  monthly_rent: number;
  bedrooms: number;
  bathrooms: number;
  bathroom_type: string;
  square_feet: number;
  has_balcony: boolean;
  in_unit_laundry: boolean;
  parking_type: string;
  pets_allowed: boolean;
  lease_term_months: number;
  furnished: boolean;
  amenities: string[];
  area: string;
  city: string;
  walk_score: number | null;
  transit_score: number | null;
}

export interface PropertiesResponse {
  properties: Property[];
  total: number;
  areas: string[];
}

// --- Tour Scheduler ---------------------------------------------------------

export interface TourAgent {
  id: string;
  name: string;
  role: string;
  /** Neighborhood names covered; [] = covers all areas. */
  areas: string[];
}

export interface Slot {
  /** Stable id = `${agent_id}|${start}`. */
  slot_id: string;
  property_id: string;
  /** ISO local "YYYY-MM-DDTHH:MM:SS". */
  start: string;
  end: string;
  agent_id: string;
  agent_name: string;
  /** Human label, e.g. "Tue Jul 22, 2:00 PM". */
  label: string;
}

export interface TourBooking {
  id: string;
  property_id: string;
  property_name: string;
  start: string;
  end: string;
  duration_minutes: number;
  agent_id: string;
  agent_name: string;
  prospect_name: string;
  prospect_email: string;
  status: "booked" | "cancelled";
  /** UTC ISO. */
  created_at: string;
  /** Self-contained "Add to Google Calendar" template link (opens a prefilled event). */
  gcal_url?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatState {
  phase:
    | "greeting"
    | "proposing"
    | "confirming"
    | "awaiting_name"
    | "booked"
    | "no_availability";
  prospect_name: string;
  last_proposed: Slot[];
  pending_slot_id: string;
}

export interface TourChatResponse {
  reply: string;
  proposed_slots: Slot[];
  booking: TourBooking | null;
  state: ChatState;
  source: "rules" | "anthropic";
}

// --- Resident Late-Payment Risk (decision-support) -------------------------

/** Risk bands. Keys stay low/medium/high; UI labels are Low / Moderate / Elevated. */
export type RiskBand = "low" | "medium" | "high";

/** One plain-English driver of the score, from TreeSHAP (or heuristic weights). */
export interface ReasonCode {
  feature: string;
  label: string;
  direction: "increases" | "decreases";
  /** Signed SHAP contribution; render |value| for the bar length. */
  contribution: number;
}

/** A single applicant's calibrated late-payment risk estimate. */
export interface RiskResult {
  applicant_id: string;
  name?: string;
  /** Calibrated P(late), 0..1. */
  probability: number;
  band: RiskBand;
  /** Top ~4 drivers by |contribution|, mixed up/down. */
  reason_codes: ReasonCode[];
  /** Low when key features (income / credit / history) are missing. */
  confidence: "high" | "low";
  /** [low, high] band from calibration spread — UI shows "~20% (12–28%)". */
  range: [number, number];
  source: "model" | "heuristic";
  model_type: "xgboost" | "histgb" | "heuristic";
  scored_at?: string;
}

/** One row in the batch risk listing. */
export interface RiskRow {
  applicant_id: string;
  name: string;
  probability: number;
  band: RiskBand;
  /** Strongest "increases" driver label. */
  top_driver: string;
}

export interface RiskListResponse {
  rows: RiskRow[];
  total: number;
  scored: number;
  avg_probability: number;
  high_risk_pct: number;
}

/** A structurally-excluded field and why it's kept out of the model. */
export interface RiskExcludedField {
  field: string;
  reason: string;
}

export interface RiskBandRange {
  band: RiskBand;
  min: number;
  max: number;
}

export interface RiskModelCard {
  name: string;
  version: string;
  trained_at: string;
  description: string;
  intended_use: string;
  features: string[];
  excluded: RiskExcludedField[];
  metrics: {
    auc: number;
    pr_auc: number;
    brier: number;
    ece: number;
    n_test: number;
    [k: string]: number;
  };
  bands: RiskBandRange[];
  limitations: string[];
  source: string;
}

/** 2x2 confusion at the operating threshold (rows = actual, cols = predicted). */
export interface RiskConfusion {
  actual_late: { pred_late: number; pred_ontime: number };
  actual_ontime: { pred_late: number; pred_ontime: number };
}

export interface RiskCalibrationBin {
  bin: number;
  predicted: number;
  observed: number;
  count: number;
}

export interface RiskSlice {
  name: string;
  n: number;
  positive_rate: number;
  auc: number;
  brier: number;
  ece: number;
  /** Set when the slice shows an elevated error / disparity worth a look. */
  flag: boolean;
}

export interface RiskEvalItem {
  id: string;
  p: number;
  band: RiskBand;
  confidence: "high" | "low";
  /** Ground-truth label: 1 = became late, 0 = on-time. */
  actual: number;
  correct: boolean;
  top_reasons: string[];
}

/** Result of POST /evals/risk/run — discrimination + calibration + fairness. */
export interface RiskEvalResult {
  auc: number;
  pr_auc: number;
  brier: number;
  ece: number;
  threshold: number;
  base_rate: number;
  n: number;
  generated_at: string;
  source: string;
  confusion: RiskConfusion;
  confusion_stats: {
    precision: number;
    recall: number;
    f1: number;
    accuracy: number;
  };
  calibration: RiskCalibrationBin[];
  slices: RiskSlice[];
  items: RiskEvalItem[];
}

// --- Risk Chat Agent (decision-support LLM chat over the risk model) --------

/** The deterministic router's chosen path for a risk-chat turn. */
export type RiskChatIntent =
  | "explain"
  | "whatif"
  | "counterfactual"
  | "compare"
  | "exclusions"
  | "general";

/** One lever moved in a what-if / counterfactual, with human labels. */
export interface RiskWhatIfChange {
  feature: string;
  label: string;
  from: number;
  to: number;
}

/** One row in a portfolio comparison (the subject or a peer/aggregate). */
export interface RiskComparisonRow {
  label: string;
  probability: number;
  band?: RiskBand;
  applicant_id?: string;
}

/**
 * A single discriminated union keyed on `kind` — the structured payload the
 * chat renders alongside its prose. `result` in score/whatif is the SAME shape
 * as GET /risk/{id}, so RiskGauge / ReasonCodes / RiskCard reuse it untransformed.
 * Phase 1 backend only emits none/score/reasons; the rest are wired for Phase 2/3.
 */
export type RiskArtifact =
  | { kind: "none" }
  | { kind: "score"; result: RiskResult }
  | { kind: "reasons"; codes: ReasonCode[] }
  | {
      kind: "whatif";
      result: RiskResult;
      baseline: number;
      changes: RiskWhatIfChange[];
    }
  | {
      kind: "counterfactual";
      target_band: RiskBand;
      achievable: boolean;
      changes: RiskWhatIfChange[];
      result?: RiskResult;
    }
  | {
      kind: "comparison";
      subject: RiskComparisonRow;
      rows: RiskComparisonRow[];
      percentile?: number;
    };

/** First SSE frame of a streamed risk-chat answer (the deterministic pass). */
export interface RiskChatMeta {
  scope: "applicant" | "portfolio";
  applicant_id: string;
  intent: RiskChatIntent;
  /** 0–3 suggested next questions. */
  follow_ups: string[];
  artifact: RiskArtifact;
}

/** Full response from POST /risk/chat (always 200; degrades server-side). */
export interface RiskChatAnswer extends RiskChatMeta {
  answer: string;
  /** "rules" => answered offline from deterministic templates (no LLM). */
  source: "anthropic" | "rules";
}

export interface RiskChatRequest {
  question: string;
  /** Omit for portfolio scope. */
  applicant_id?: string;
  history?: { role: string; content: string }[];
}

export interface DashboardStats {
  applicants_total: number;
  verdicts: { qualified: number; needs_review: number; not_qualified: number };
  recent_applicants: Array<{
    id: string;
    name: string;
    created_at: string;
    verdict: "qualified" | "needs_review" | "not_qualified";
  }>;
  properties: {
    total: number;
    rent_min: number;
    rent_max: number;
    pets_allowed: number;
    areas: number;
  };
  traffic: {
    events_total: number;
    avg_latency_ms_by_endpoint: Record<string, number>;
    faithfulness_violations: number;
  };
  feedback: { up: number; down: number };
}

// --- Residents: current-resident risk assessment (decision-support) --------
//
// A NEW feature, separate from applicant-screening Risk above. Forward-looking
// risk on CURRENT residents from 5 years (60 months) of rent-ledger history.
// Four independent predictions per resident (late / arrears / churn / serious),
// each degrading to a transparent heuristic. Reuses RiskBand + ReasonCode.

/** One month's status in a resident's rent ledger. */
export type LedgerStatus = "paid" | "paid_late" | "partial" | "missed";

/**
 * Churn applicability. Churn is only labeled/scored for leases ending within
 * the 6-month horizon; otherwise the API returns "not_applicable".
 */
export type ChurnBand = RiskBand | "not_applicable";

/** One month of the 60-month rent ledger (oldest → newest, ending at snapshot). */
export interface LedgerEntry {
  /** "YYYY-MM". */
  period: string;
  rent_charged: number;
  amount_paid: number;
  /** ISO date the payment posted; null when nothing was paid that month. */
  paid_date?: string | null;
  days_late: number;
  late_fee: number;
  on_time: boolean;
  status: LedgerStatus;
  /** Rolling balance owed after this month's activity. */
  balance_after: number;
  notices_sent: number;
  notice_responded: number;
}

/**
 * A full committed resident record: immutable facts + the 60-month ledger.
 * Time-relative stats (current_balance / tenure / arrears / response rate) are
 * derived at scoring; the detail endpoint may echo them back as optional fields.
 */
export interface Resident {
  resident_id: string;
  property_id: string;
  unit_id: string;
  unit_bedrooms: number;
  base_rent: number;
  // lease
  lease_start: string;
  lease_end: string;
  lease_term_months: number;
  renewal_offer_sent: boolean;
  autopay_enrolled: boolean;
  deposit_held: number;
  move_in_date: string;
  prior_renewals: number;
  // financial (snapshot)
  monthly_income: number;
  other_income_monthly: number;
  income_verified: boolean;
  // engagement (stored counts)
  maintenance_requests_12mo: number;
  complaints_12mo: number;
  portal_logins_90d: number;
  // history
  ledger: LedgerEntry[];
  dgp_version?: string;
  // derived-at-scoring stats (present on the detail response; optional)
  tenure_months?: number;
  current_balance?: number;
  expected_arrears?: number;
  notice_response_rate?: number;
  on_time_streak_months?: number;
  [k: string]: unknown;
}

/** One row in the portfolio / property resident listing. */
export interface ResidentRow {
  resident_id: string;
  property_id: string;
  unit_id: string;
  base_rent: number;
  tenure_months: number;
  /** Calibrated P(any late payment next quarter), 0..1. */
  late_probability: number;
  late_band: RiskBand;
  /** Expected $ balance owed at the end of next quarter. */
  expected_arrears: number;
  /** null when the lease isn't ending within the churn horizon. */
  churn_probability: number | null;
  churn_status: ChurnBand;
  serious_probability: number;
  serious_band: RiskBand;
  current_balance: number;
  /** Strongest driver label for a compact table cell. */
  top_driver: string;
}

export interface ResidentListResponse {
  residents: ResidentRow[];
  count: number;
  property_id?: string | null;
  source: string;
}

/** Binary-classifier prediction (late / serious) — shape reused by RiskGauge. */
export interface ResidentClassPrediction {
  probability: number;
  band: RiskBand;
  /** [low, high] calibration spread — UI shows "~20% (12–28%)". */
  range: [number, number];
  reason_codes: ReasonCode[];
  confidence: "high" | "low";
  source: string;
  model_type: string;
}

/** Serious-delinquency prediction always routes to a human reviewer. */
export interface ResidentSeriousPrediction extends ResidentClassPrediction {
  routes_to_review: boolean;
}

/** Expected-arrears regression prediction ($ owed next quarter). */
export interface ResidentArrearsPrediction {
  expected_balance: number;
  /** Prediction interval [low, high] in dollars. */
  interval: [number, number];
  reason_codes: ReasonCode[];
  confidence: "high" | "low";
  source: string;
}

/** Churn prediction — null / "not_applicable" when the lease isn't ending soon. */
export interface ResidentChurnPrediction {
  probability: number | null;
  band: ChurnBand;
  months_to_lease_end: number;
  reason_codes: ReasonCode[];
  confidence: "high" | "low";
  source: string;
}

/** The four forward-looking predictions for one resident. */
export interface ResidentPredictions {
  late: ResidentClassPrediction;
  arrears: ResidentArrearsPrediction;
  churn: ResidentChurnPrediction;
  serious: ResidentSeriousPrediction;
}

/** GET /residents/{id} — full resident record + the four predictions. */
export interface ResidentDetail {
  resident: Resident;
  predictions: ResidentPredictions;
}

/** Per-property rollup used by the selector and the property drill-down. */
export interface PropertyResidentRollup {
  property_id?: string;
  name?: string;
  count: number;
  predicted_late_rate?: number;
  total_expected_arrears?: number;
  churn_risk_count?: number;
  serious_flag_count?: number;
  [k: string]: unknown;
}

/** GET /properties/{id}/residents. */
export interface PropertyResidentsResponse {
  residents: ResidentRow[];
  rollup: PropertyResidentRollup;
}

/** One property's aggregates in the portfolio summary. */
export interface PortfolioProperty {
  property_id: string;
  name?: string;
  resident_count: number;
  predicted_late_rate: number;
  total_expected_arrears: number;
  churn_risk_count: number;
  serious_flag_count: number;
  late_bands?: Record<string, number>;
  serious_bands?: Record<string, number>;
  churn_bands?: Record<string, number>;
  [k: string]: unknown;
}

/** GET /residents/portfolio/summary. */
export interface PortfolioSummary {
  properties: PortfolioProperty[];
  overall: {
    resident_count: number;
    predicted_late_rate: number;
    total_expected_arrears: number;
    churn_risk_count: number;
    serious_flag_count: number;
    late_bands?: Record<string, number>;
    [k: string]: number | Record<string, number> | undefined;
  };
}

/** One target's governance block inside the resident model card. */
export interface ResidentTargetCard {
  feature_order?: string[];
  features?: string[];
  metrics?: Record<string, number>;
  bands?: RiskBandRange[];
  model_type?: string;
  source?: string;
  [k: string]: unknown;
}

/**
 * GET /residents/model-card. Multi-target (late / arrears / churn / serious)
 * plus the shared structurally-excluded fields. Typed loosely and rendered
 * defensively — it's best-effort, like the Risk model card.
 */
export interface ResidentModelCard {
  name?: string;
  version?: string;
  dgp_version?: string;
  generated_at?: string;
  trained_at?: string;
  description?: string;
  intended_use?: string;
  /** Keyed by target name: late / arrears / churn / serious. */
  targets?: Record<string, ResidentTargetCard>;
  excluded?: RiskExcludedField[];
  limitations?: string[];
  source?: string;
  [k: string]: unknown;
}
