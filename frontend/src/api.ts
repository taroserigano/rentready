import type {
  ApplicantProfile,
  ApplicantSummary,
  AskResponse,
  CandidatesResponse,
  DashboardStats,
  DecisionItem,
  EligibilityResult,
  GoalSeekResult,
  PropertiesResponse,
  RecommendResponse,
  ScreenResult,
  SimulateResponse,
  StrengthResult,
  UploadResponse,
  TourAgent,
  Slot,
  TourBooking,
  ChatMessage,
  ChatState,
  TourChatResponse,
  ConciergeAnswer,
  ConciergeMeta,
  ConciergeEvalResult,
  LeaseDoc,
  RiskResult,
  RiskListResponse,
  RiskModelCard,
  RiskEvalResult,
  RiskChatRequest,
  RiskChatAnswer,
  RiskChatMeta,
  ResidentListResponse,
  ResidentDetail,
  ResidentPredictions,
  PropertyResidentsResponse,
  ResidentPropertiesResponse,
  PortfolioSummary,
  ResidentModelCard,
} from "./types";

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

/** URL of the standalone, printable HTML evaluation report (latest run). */
export const evalReportUrl = `${BASE}/evals/report`;

/** URL of an applicant's application PDF (served inline). */
export const pdfUrl = (applicantId: string) =>
  `${BASE}/applicants/${applicantId}/pdf`;

/** URL of a property's generated lease agreement, as a real PDF (served inline). */
export const leasePdfUrl = (propertyId: string) =>
  `${BASE}/concierge/lease/${encodeURIComponent(propertyId)}/pdf`;

/** URL that downloads a tour booking as an .ics calendar file. */
export const toursIcsUrl = (bookingId: string) =>
  `${BASE}/tours/${bookingId}/calendar.ics`;

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    let message = `${res.status}: ${text}`;
    try {
      // FastAPI errors look like {"detail": "..."} — surface the plain
      // message so components can show it to the user directly.
      const detail = JSON.parse(text)?.detail;
      if (typeof detail === "string" && detail) message = detail;
    } catch {
      // body was not JSON; keep the raw text
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export async function uploadPdf(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return json(await fetch(`${BASE}/upload`, { method: "POST", body: form }));
}

export async function getEligibility(id: string): Promise<EligibilityResult> {
  return json(await fetch(`${BASE}/eligibility/${id}`));
}

export async function getRecommendations(
  id: string,
): Promise<RecommendResponse> {
  return json(await fetch(`${BASE}/recommend/${id}`));
}

export async function simulate(
  applicant_id: string,
  overrides: {
    monthly_income?: number;
    desired_rent?: number;
    credit_score?: number;
  },
): Promise<SimulateResponse> {
  return json(
    await fetch(`${BASE}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ applicant_id, ...overrides }),
    }),
  );
}

export async function ask(
  applicant_id: string,
  question: string,
): Promise<AskResponse> {
  return json(
    await fetch(`${BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ applicant_id, question }),
    }),
  );
}

export async function getHealth(): Promise<Record<string, unknown>> {
  return json(await fetch(`${BASE}/health`));
}

export async function applyForm(
  profile: ApplicantProfile,
): Promise<UploadResponse> {
  return json(
    await fetch(`${BASE}/apply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(profile),
    }),
  );
}

export async function listApplicants(): Promise<ApplicantSummary[]> {
  return json(await fetch(`${BASE}/applicants`));
}

export async function getApplicant(
  id: string,
): Promise<{ applicant_id: string; profile: ApplicantProfile }> {
  return json(await fetch(`${BASE}/applicants/${id}`));
}

export async function deleteApplicant(
  id: string,
): Promise<{ deleted: string }> {
  return json(await fetch(`${BASE}/applicants/${id}`, { method: "DELETE" }));
}

export async function getProperties(
  params: {
    area?: string;
    max_rent?: number;
    min_bedrooms?: number;
    pets_allowed?: boolean;
    q?: string;
  } = {},
): Promise<PropertiesResponse> {
  const qs = new URLSearchParams();
  if (params.area) qs.set("area", params.area);
  if (params.max_rent != null) qs.set("max_rent", String(params.max_rent));
  if (params.min_bedrooms != null)
    qs.set("min_bedrooms", String(params.min_bedrooms));
  if (params.pets_allowed) qs.set("pets_allowed", "true");
  if (params.q) qs.set("q", params.q);
  const suffix = qs.toString() ? `?${qs.toString()}` : "";
  return json(await fetch(`${BASE}/properties${suffix}`));
}

export async function getDashboardStats(): Promise<DashboardStats> {
  return json(await fetch(`${BASE}/dashboard/stats`));
}

export interface Sample {
  slug: string;
  name: string;
}

export async function getSamples(): Promise<Sample[]> {
  return json(await fetch(`${BASE}/samples`));
}

export async function loadSample(slug: string): Promise<UploadResponse> {
  return json(await fetch(`${BASE}/samples/${slug}`, { method: "POST" }));
}

export interface EvalSuite {
  id: string;
  name: string;
  description: string;
  metric: string;
}

export async function getEvalSuites(): Promise<EvalSuite[]> {
  return json(await fetch(`${BASE}/evals/suites`));
}

export async function getEvalsLatest(): Promise<Record<string, any>> {
  return json(await fetch(`${BASE}/evals/latest`));
}

export async function runEvals(): Promise<Record<string, any>> {
  return json(await fetch(`${BASE}/evals/run`, { method: "POST" }));
}

export async function runEvalsDeterministic(): Promise<Record<string, any>> {
  return json(await fetch(`${BASE}/evals/run-deterministic`, { method: "POST" }));
}

export interface EvalSnapshot {
  generated_at: string;
  eligibility_accuracy: number | null;
  extraction_field_accuracy: number | null;
  ndcg_at_5: number | null;
  judge_groundedness_pct: number | null;
  ragas_faithfulness: number | null;
  ragas_answer_correctness: number | null;
}

export async function getEvalHistory(limit = 30): Promise<EvalSnapshot[]> {
  return json(await fetch(`${BASE}/evals/history?limit=${limit}`));
}

export async function runJudge(): Promise<Record<string, any>> {
  return json(await fetch(`${BASE}/evals/judge`, { method: "POST" }));
}

export async function runRagas(): Promise<Record<string, any>> {
  return json(await fetch(`${BASE}/evals/ragas`, { method: "POST" }));
}

export async function runLangsmith(): Promise<Record<string, any>> {
  return json(await fetch(`${BASE}/evals/langsmith`, { method: "POST" }));
}

export interface AbVariant {
  id: string;
  label: string;
  model: string;
}

export async function getAbVariants(): Promise<AbVariant[]> {
  return json(await fetch(`${BASE}/evals/ab/variants`));
}

export async function runAb(a: string, b: string): Promise<Record<string, any>> {
  return json(
    await fetch(`${BASE}/evals/ab/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a, b }),
    }),
  );
}

export async function getMonitoring(): Promise<Record<string, any>> {
  return json(await fetch(`${BASE}/monitoring/overview`));
}

export async function pushDatadog(): Promise<Record<string, any>> {
  return json(await fetch(`${BASE}/monitoring/push-datadog`, { method: "POST" }));
}

export async function sendFeedback(input: {
  applicant_id?: string;
  target: string;
  rating: "up" | "down";
  item_id?: string;
  comment?: string;
}): Promise<{ ok: boolean }> {
  return json(
    await fetch(`${BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function graphAsk(
  question: string,
): Promise<{ answer: string; cypher: string }> {
  return json(
    await fetch(`${BASE}/graph-ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
  );
}

// --- Landlord-side screening (F1/F2) + goal-seek (F10) --------------------

export async function screenApplicant(
  propertyId: string,
  applicantId: string,
): Promise<ScreenResult> {
  return json(
    await fetch(`${BASE}/properties/${propertyId}/screen?applicant_id=${applicantId}`),
  );
}

export async function getPropertyCandidates(
  propertyId: string,
  limit = 10,
): Promise<CandidatesResponse> {
  return json(
    await fetch(`${BASE}/properties/${propertyId}/candidates?limit=${limit}`),
  );
}

export async function goalSeek(
  applicantId: string,
  solveFor: "monthly_income" | "desired_rent",
): Promise<GoalSeekResult> {
  return json(
    await fetch(`${BASE}/simulate/goal-seek`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ applicant_id: applicantId, solve_for: solveFor }),
    }),
  );
}

// --- Strength (F4) + decisions (F6) ---------------------------------------

export async function getStrength(applicantId: string): Promise<StrengthResult> {
  return json(await fetch(`${BASE}/applicants/${applicantId}/strength`));
}

export async function postDecision(
  applicantId: string,
  body: { action: string; note?: string; reviewer?: string },
): Promise<{ ok: boolean; status: string }> {
  return json(
    await fetch(`${BASE}/applicants/${applicantId}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export async function getDecisions(
  applicantId: string,
): Promise<{ applicant_id: string; decisions: DecisionItem[] }> {
  return json(await fetch(`${BASE}/applicants/${applicantId}/decisions`));
}

// --- Tour Scheduler --------------------------------------------------------

export async function getTourStaff(propertyId?: string): Promise<TourAgent[]> {
  const suffix = propertyId ? `?property_id=${encodeURIComponent(propertyId)}` : "";
  const res = await json<{ staff: TourAgent[]; total: number }>(
    await fetch(`${BASE}/tours/staff${suffix}`),
  );
  return res.staff;
}

export async function getOpenSlots(
  propertyId: string,
  opts: { from?: string; to?: string; timeOfDay?: string } = {},
): Promise<Slot[]> {
  const qs = new URLSearchParams({ property_id: propertyId });
  if (opts.from) qs.set("date_from", opts.from);
  if (opts.to) qs.set("date_to", opts.to);
  if (opts.timeOfDay) qs.set("time_of_day", opts.timeOfDay);
  const res = await json<{ property_id: string; slots: Slot[]; total: number }>(
    await fetch(`${BASE}/tours/slots?${qs.toString()}`),
  );
  return res.slots;
}

export async function tourChat(input: {
  messages: ChatMessage[];
  property_id?: string;
  state?: ChatState;
  selected_slot_id?: string;
}): Promise<TourChatResponse> {
  return json(
    await fetch(`${BASE}/tours/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function bookTour(input: {
  property_id: string;
  slot_id: string;
  prospect_name: string;
  prospect_email?: string;
}): Promise<TourBooking> {
  const res = await json<{ ok: boolean; booking: TourBooking }>(
    await fetch(`${BASE}/tours/book`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
  return res.booking;
}

export async function listTours(
  filters: {
    email?: string;
    applicant_id?: string;
    property_id?: string;
    status?: string;
  } = {},
): Promise<TourBooking[]> {
  const qs = new URLSearchParams();
  if (filters.email) qs.set("email", filters.email);
  if (filters.applicant_id) qs.set("applicant_id", filters.applicant_id);
  if (filters.property_id) qs.set("property_id", filters.property_id);
  if (filters.status) qs.set("status", filters.status);
  const suffix = qs.toString() ? `?${qs.toString()}` : "";
  const res = await json<{ tours: TourBooking[]; total: number }>(
    await fetch(`${BASE}/tours${suffix}`),
  );
  return res.tours;
}

export async function cancelTour(
  bookingId: string,
): Promise<{ status: string }> {
  return json(
    await fetch(`${BASE}/tours/${bookingId}`, { method: "DELETE" }),
  );
}

// --- Property & Lease Concierge --------------------------------------------

export async function conciergeAsk(input: {
  question: string;
  property_id?: string;
  history?: { role: string; content: string }[];
}): Promise<ConciergeAnswer> {
  return json(
    await fetch(`${BASE}/concierge/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

/** Callbacks driven by the SSE stream from POST /concierge/ask/stream. */
export interface ConciergeStreamHandlers {
  onMeta: (meta: ConciergeMeta) => void;
  onToken: (text: string) => void;
  onDone: (source: string) => void;
}

/**
 * Stream a concierge answer over SSE. The deterministic retrieval pass arrives
 * first as a `meta` frame; the LLM prose then streams as `token` frames; a
 * final `done` frame carries the answer's source. Frames are `data: <json>\n\n`.
 * Throws on a network/transport error so the caller can fall back to the
 * non-streaming `conciergeAsk`.
 */
export async function conciergeAskStream(
  input: {
    question: string;
    property_id?: string;
    history?: { role: string; content: string }[];
  },
  handlers: ConciergeStreamHandlers,
): Promise<void> {
  const res = await fetch(`${BASE}/concierge/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok || !res.body) {
    throw new Error(`stream failed: ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (raw: string) => {
    // A frame may carry several lines; only the `data:` payload is JSON.
    const line = raw
      .split("\n")
      .find((l) => l.startsWith("data:"));
    if (!line) return;
    const payload = line.slice(5).trim();
    if (!payload) return;
    let frame: any;
    try {
      frame = JSON.parse(payload);
    } catch {
      return; // ignore malformed frames rather than crashing the stream
    }
    if (frame.type === "meta") {
      handlers.onMeta({
        route: frame.route ?? "general",
        property_id: frame.property_id ?? "",
        sources: frame.sources ?? [],
        follow_ups: frame.follow_ups ?? [],
        comparison: frame.comparison ?? [],
      });
    } else if (frame.type === "token") {
      handlers.onToken(frame.text ?? "");
    } else if (frame.type === "done") {
      handlers.onDone(frame.source ?? "anthropic");
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      if (frame.trim()) dispatch(frame);
    }
  }
  // Flush any trailing frame that arrived without a terminating blank line.
  if (buffer.trim()) dispatch(buffer);
}

export async function runConciergeEval(
  useLlm = false,
): Promise<ConciergeEvalResult> {
  return json(
    await fetch(`${BASE}/evals/concierge/run?use_llm=${useLlm}`, {
      method: "POST",
    }),
  );
}

/** Last persisted concierge-eval result, or null if it's never been run. */
export async function getConciergeEvalLatest(): Promise<ConciergeEvalResult | null> {
  const res = await fetch(`${BASE}/evals/concierge/latest`);
  if (!res.ok) return null;
  const data = (await res.json()) as ConciergeEvalResult | null;
  return data && typeof data.n === "number" ? data : null;
}

// --- Resident Late-Payment Risk (decision-support) -------------------------

/** Calibrated late-payment risk for a saved applicant (404 if unknown). */
export async function getRisk(id: string): Promise<RiskResult> {
  return json(await fetch(`${BASE}/risk/${encodeURIComponent(id)}`));
}

/** Score an ad-hoc profile without persisting it. */
export async function scoreRisk(profile: ApplicantProfile): Promise<RiskResult> {
  return json(
    await fetch(`${BASE}/risk/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile }),
    }),
  );
}

/** Batch risk over all saved applicants, ranked highest-first. */
export async function listRisk(): Promise<RiskListResponse> {
  return json(await fetch(`${BASE}/risk`));
}

export async function getRiskModelCard(): Promise<RiskModelCard> {
  return json(await fetch(`${BASE}/risk/model-card`));
}

export async function runRiskEval(): Promise<RiskEvalResult> {
  return json(await fetch(`${BASE}/evals/risk/run`, { method: "POST" }));
}

/** Last persisted risk-eval result, or null if it's never been run. */
export async function getRiskEvalLatest(): Promise<RiskEvalResult | null> {
  const res = await fetch(`${BASE}/evals/risk/latest`);
  if (!res.ok) return null;
  const data = (await res.json()) as RiskEvalResult | null;
  return data && typeof data.n === "number" ? data : null;
}

// --- Risk Chat Agent -------------------------------------------------------

/**
 * Ask the risk chat agent (non-streaming). Always resolves 200 — the backend
 * degrades to deterministic rules (source="rules") when no LLM is available.
 */
export async function askRiskChat(
  input: RiskChatRequest,
): Promise<RiskChatAnswer> {
  return json(
    await fetch(`${BASE}/risk/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

/** Callbacks driven by the SSE stream from POST /risk/chat/stream. */
export interface RiskChatStreamHandlers {
  onMeta: (meta: RiskChatMeta) => void;
  onToken: (text: string) => void;
  onDone: (source: string) => void;
}

/**
 * Stream a risk-chat answer over SSE. The deterministic router pass arrives
 * first as a `meta` frame (so the artifact/gauge can paint before prose); the
 * LLM prose then streams as `token` frames; a final `done` frame carries the
 * answer's source. Frames are `data: <json>\n\n`. Throws on a network/transport
 * error so the caller can fall back to the non-streaming `askRiskChat`.
 *
 * NOTE: the /risk/chat/stream endpoint does not exist until Phase 3 — this is
 * provided now to avoid churn but is not yet wired into the rail.
 */
export async function askRiskChatStream(
  input: RiskChatRequest,
  handlers: RiskChatStreamHandlers,
): Promise<void> {
  const res = await fetch(`${BASE}/risk/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok || !res.body) {
    throw new Error(`stream failed: ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (raw: string) => {
    // A frame may carry several lines; only the `data:` payload is JSON.
    const line = raw
      .split("\n")
      .find((l) => l.startsWith("data:"));
    if (!line) return;
    const payload = line.slice(5).trim();
    if (!payload) return;
    let frame: any;
    try {
      frame = JSON.parse(payload);
    } catch {
      return; // ignore malformed frames rather than crashing the stream
    }
    if (frame.type === "meta") {
      handlers.onMeta({
        scope: frame.scope ?? "portfolio",
        applicant_id: frame.applicant_id ?? "",
        intent: frame.intent ?? "general",
        follow_ups: frame.follow_ups ?? [],
        artifact: frame.artifact ?? { kind: "none" },
      });
    } else if (frame.type === "token") {
      handlers.onToken(frame.text ?? "");
    } else if (frame.type === "done") {
      handlers.onDone(frame.source ?? "anthropic");
    }
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      if (frame.trim()) dispatch(frame);
    }
  }
  // Flush any trailing frame that arrived without a terminating blank line.
  if (buffer.trim()) dispatch(buffer);
}

// --- Residents: current-resident risk assessment (decision-support) --------

/**
 * Portfolio-wide resident listing, ranked rows. Pass a `propertyId` to scope to
 * one property. Always resolves 200 (the backend degrades gracefully).
 */
export async function listResidents(
  propertyId?: string,
): Promise<ResidentListResponse> {
  const suffix = propertyId
    ? `?property_id=${encodeURIComponent(propertyId)}`
    : "";
  return json(await fetch(`${BASE}/residents${suffix}`));
}

/** One resident's full record + the four predictions (404 if unknown). */
export async function getResident(id: string): Promise<ResidentDetail> {
  return json(await fetch(`${BASE}/residents/${encodeURIComponent(id)}`));
}

/** Residents for one property + that property's rollup. */
export async function listPropertyResidents(
  propertyId: string,
): Promise<PropertyResidentsResponse> {
  return json(
    await fetch(`${BASE}/properties/${encodeURIComponent(propertyId)}/residents`),
  );
}

/** Cheap property picker (id, name, headcount) — no scoring. */
export async function getResidentProperties(): Promise<ResidentPropertiesResponse> {
  return json(await fetch(`${BASE}/residents/properties`));
}

/** Per-property + overall portfolio rollups (KPI tiles + selector). */
export async function getResidentsPortfolio(): Promise<PortfolioSummary> {
  return json(await fetch(`${BASE}/residents/portfolio/summary`));
}

export async function getResidentModelCard(): Promise<ResidentModelCard> {
  return json(await fetch(`${BASE}/residents/model-card`));
}

/** Re-score one resident on demand; returns the four predictions. */
export async function scoreResident(
  id: string,
): Promise<ResidentPredictions> {
  return json(
    await fetch(`${BASE}/residents/${encodeURIComponent(id)}/score`, {
      method: "POST",
    }),
  );
}

export async function getLease(propertyId: string): Promise<LeaseDoc> {
  return json(
    await fetch(`${BASE}/concierge/lease/${encodeURIComponent(propertyId)}`),
  );
}

export async function conciergeStatus(): Promise<{
  indexed: number;
  collection: string;
}> {
  return json(await fetch(`${BASE}/concierge/status`));
}
