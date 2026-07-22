import type { ChurnBand, HealthGrade, LedgerStatus, RiskBand } from "../../types";

/**
 * Residents tone helpers — mirror the Risk page's riskTone conventions so the
 * two features read as one system. Meaning never rests on colour alone: every
 * band/status carries a label, and callers pair the tone class with text.
 */

/** UI labels for the risk bands (keys stay low/medium/high). */
export { BAND_LABEL } from "../risk/riskTone";

/** Badge / meter tone-class suffix per band — same mapping the Risk page uses. */
export { BAND_TONE } from "../risk/riskTone";

/** "~20% (12–28%)" summary from a probability + calibration range. */
export { formatProbRange } from "../risk/riskTone";

import { BAND_LABEL, BAND_TONE } from "../risk/riskTone";

/** Churn band → label, including the "not applicable" sentinel. */
export function churnLabel(band: ChurnBand): string {
  return band === "not_applicable" ? "Not applicable" : BAND_LABEL[band];
}

/** Churn band → tone suffix; "not applicable" is neutral (info). */
export function churnTone(band: ChurnBand): "good" | "warn" | "bad" | "info" {
  return band === "not_applicable" ? "info" : BAND_TONE[band];
}

/** Human label for a resident-risk band in a given prediction context. */
export function bandLabel(band: RiskBand): string {
  return BAND_LABEL[band];
}

/**
 * Resident-specific decision-support disclaimer (the applicant model's wording
 * about "the application" / "lease" doesn't fit current residents). Passed to
 * <RiskDisclaimer text=…>.
 */
export const RESIDENT_DISCLAIMER =
  "Decision-support only — not an automated decision. Forward-looking estimates " +
  "from a model trained on synthetic data, meant to help a person prioritise " +
  "proactive outreach and retention. They never trigger eviction, denial, " +
  "pricing, or any automated action; serious-delinquency flags route to a human " +
  "reviewer, and the model never uses race, national origin, sex, familial " +
  "status, disability, age, or location.";

/**
 * Payment-status → presentation. `tone` is the semantic bucket; `token` is the
 * CSS custom property that colours the timeline cell (NO hardcoded colours);
 * `label` is the human string used in the legend and hover tooltip.
 *
 * The four chart tokens are documented as colourblind-safe by hue + lightness,
 * and every cell also exposes its status as text — colour is never the only cue.
 */
export interface StatusStyle {
  label: string;
  /** CSS var reference for the cell fill. */
  token: string;
  /** Semantic bucket, for optional badge tone reuse. */
  tone: "good" | "warn" | "bad" | "info";
  /** Timeline cell modifier class (see index.css `.pay-cell.is-*`). */
  cls: string;
}

export const STATUS_STYLE: Record<LedgerStatus, StatusStyle> = {
  paid: { label: "Paid on time", token: "var(--chart-2)", tone: "good", cls: "is-paid" },
  paid_late: { label: "Paid late", token: "var(--chart-3)", tone: "warn", cls: "is-late" },
  partial: { label: "Partial", token: "var(--chart-5)", tone: "info", cls: "is-partial" },
  missed: { label: "Missed", token: "var(--chart-4)", tone: "bad", cls: "is-missed" },
};

/** Ordered legend entries for the timeline key. */
export const STATUS_ORDER: LedgerStatus[] = [
  "paid",
  "paid_late",
  "partial",
  "missed",
];

/** Compact USD, e.g. "$1,240" (no cents) — used across tiles and tables. */
export function usd(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

/** "42%" from a 0..1 probability; em dash for null. */
export function pct(p: number | null | undefined): string {
  if (p == null || Number.isNaN(p)) return "—";
  return `${Math.round(p * 100)}%`;
}

// --- Property health --------------------------------------------------------

/**
 * Letter grade → tone-class suffix (same good/warn/bad vocabulary as the risk
 * bands, so the ranking's meters and chips match the rest of the app). Health
 * is "higher is better", so A/B read good, C warns, D/F read bad. Meaning never
 * rests on colour — the grade letter and score always accompany the tone.
 */
export function gradeTone(
  grade: HealthGrade | string,
): "good" | "warn" | "bad" | "info" {
  switch (grade) {
    case "A":
    case "B":
      return "good";
    case "C":
      return "warn";
    case "D":
    case "F":
      return "bad";
    default:
      return "info";
  }
}

// --- Multi-head labels ------------------------------------------------------

/** Family → section heading in the resident detail. */
export const FAMILY_LABEL: Record<string, string> = {
  late: "Late-payment outlook",
  frequency: "How often",
  severity: "How severe",
  arrears: "Arrears ($)",
  cure: "Getting current",
  retention: "Retention",
};

/** Family → one-line supporting description. */
export const FAMILY_HINT: Record<string, string> = {
  late: "Chance of any late/partial/missed payment across each horizon.",
  frequency: "Expected number of trouble months over the next year.",
  severity: "How bad lateness could get — days late and delinquency depth.",
  arrears: "Projected dollar balance owed at each horizon.",
  cure: "Whether and when an outstanding balance clears.",
  retention: "Chance the resident does not renew (renewal window only).",
};

/** Head name → short human label used on tiles/gauges. */
export const HEAD_LABEL: Record<string, string> = {
  late_1m: "Next month",
  late_3m: "Next quarter",
  late_6m: "Next 6 months",
  late_12m: "Next year",
  late_count_12m: "Late months",
  missed_count_12m: "Missed payments",
  max_days_late_12m: "Max days late",
  p_30d_12m: "30+ days late",
  p_60d_12m: "60+ days late",
  p_90d_12m: "90+ days late",
  delinquency_bucket_12m: "Worst delinquency",
  serious: "Serious delinquency",
  arrears_3m: "In 3 months",
  arrears_12m: "In 12 months",
  peak_balance_12m: "Peak balance",
  p_cure_6m: "Clears in 6 months",
  months_to_cure: "Time to clear",
  churn: "Non-renewal (≤6mo)",
  churn_12m: "Non-renewal (≤12mo)",
};

/** Fall back to a de-slugged label for any head the map doesn't name. */
export function headLabel(name: string): string {
  return HEAD_LABEL[name] ?? name.replace(/_/g, " ");
}
