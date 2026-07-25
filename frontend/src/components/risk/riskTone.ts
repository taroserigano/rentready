import type { RiskBand } from "../../types";

/** UI labels for the bands (keys stay low/medium/high for the tone map). */
export const BAND_LABEL: Record<RiskBand, string> = {
  low: "Low",
  medium: "Moderate",
  high: "Elevated",
};

/** Badge / verdict tone class suffix. */
export const BAND_TONE: Record<RiskBand, "good" | "warn" | "bad"> = {
  low: "good",
  medium: "warn",
  high: "bad",
};

/** Chart fill token per band (matches the Aurora chart ramp). */
export const BAND_COLOR: Record<RiskBand, string> = {
  low: "var(--chart-2)",
  medium: "var(--chart-3)",
  high: "var(--chart-4)",
};

/** `.meter > i` tone class per band. */
export const BAND_METER: Record<RiskBand, "good" | "warn" | "bad"> = {
  low: "good",
  medium: "warn",
  high: "bad",
};

/**
 * Mandatory decision-support framing. Rendered on the Risk page banner and
 * inline on every RiskCard. Kept single-sourced so the copy never drifts.
 */
export const RISK_DISCLAIMER =
  "Decision-support only — not an automated decision. An estimated probability from a model " +
  "trained on synthetic data, to help a person review the application. It does not approve, deny, " +
  "price, or condition any lease. No protected attribute — race, national origin, sex, familial " +
  "status, disability, age — or location field is a model input; dollar-denominated balance " +
  "features can carry residual property-scale information, which is audited by slicing.";

/** "~20% (12–28%)" style summary from a probability and its calibration range. */
export function formatProbRange(probability: number, range: [number, number]): string {
  const p = Math.round(probability * 100);
  const lo = Math.round(range[0] * 100);
  const hi = Math.round(range[1] * 100);
  return `~${p}% (${lo}–${hi}%)`;
}
