import type { RiskBand } from "../../types";
import { BAND_LABEL } from "./riskTone";

/** The band one step lower than `band` (the realistic counterfactual target). */
const NEXT_LOWER: Partial<Record<RiskBand, RiskBand>> = {
  high: "medium",
  medium: "low",
};

/**
 * Suggested opening questions for the risk chat, seeded when the thread is
 * empty. Pure — no I/O. Applicant scope is personalized by `name`; the
 * counterfactual starter is band-gated (only offered when there's a lower band
 * to move toward, e.g. "What would move them to Moderate?").
 */
export function riskStarters(
  name: string | undefined,
  band: RiskBand | undefined,
  scope: "applicant" | "portfolio",
): string[] {
  if (scope === "portfolio") {
    return [
      "How is late-payment risk distributed across applicants?",
      "How many applicants are Elevated risk?",
      "Which factors are excluded from the model?",
      "What do the risk bands mean?",
    ];
  }

  const who = name && name.trim() ? name.trim() : "this applicant";
  const starters = [
    `Why is ${who} scored this way?`,
    `What's driving ${who}'s risk?`,
  ];

  const target = band ? NEXT_LOWER[band] : undefined;
  if (target) {
    starters.push(`What would move them to ${BAND_LABEL[target]}?`);
  }

  starters.push(`How does ${who} compare to the portfolio?`);
  starters.push("Which factors are excluded from this model?");
  return starters;
}
