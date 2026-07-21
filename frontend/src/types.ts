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
