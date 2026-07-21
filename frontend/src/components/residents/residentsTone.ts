import type { ChurnBand, LedgerStatus, RiskBand } from "../../types";

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
