/**
 * Regression tests for defects found in the full-app evaluation.
 *
 * Each block names the bug it locks down. All of these shipped to a live
 * public deployment with 370 backend + 95 frontend tests passing, which is
 * precisely why they're worth pinning.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RiskDisclaimer } from "../components/risk/RiskCard";
import { ResidentModelCard } from "../components/residents/ResidentModelCard";
import { LateCountBreakdownChart } from "../components/residents/LateCountBreakdownChart";
import { RISK_DISCLAIMER } from "../components/risk/riskTone";
import { RESIDENT_DISCLAIMER } from "../components/residents/residentsTone";
import type { ResidentModelCard as ResidentModelCardT } from "../types";

describe("RiskDisclaimer (was exported but rendered NOWHERE in the app)", () => {
  it("renders the risk copy as a banner", () => {
    render(<RiskDisclaimer variant="banner" />);
    expect(screen.getByText(/Decision-support only/)).toBeInTheDocument();
  });

  it("renders the risk copy inline by default", () => {
    render(<RiskDisclaimer />);
    expect(screen.getByText(/Decision-support only/)).toBeInTheDocument();
  });

  it("honours the `text` prop in the INLINE variant", () => {
    // The inline branch hardcoded RISK_DISCLAIMER and ignored `text`, so the
    // resident wording could never render even when passed.
    render(<RiskDisclaimer variant="inline" text={RESIDENT_DISCLAIMER} />);
    expect(screen.getByText(RESIDENT_DISCLAIMER)).toBeInTheDocument();
    expect(screen.queryByText(RISK_DISCLAIMER)).not.toBeInTheDocument();
  });

  it("honours the `text` prop in the BANNER variant", () => {
    render(<RiskDisclaimer variant="banner" text={RESIDENT_DISCLAIMER} />);
    expect(screen.getByText(RESIDENT_DISCLAIMER)).toBeInTheDocument();
  });

  it("does not overstate the fairness guarantee", () => {
    // "never uses ... location" was not honest: dollar-denominated balance
    // features carry residual property-scale (hence city) information.
    expect(RISK_DISCLAIMER).not.toMatch(/never uses .*location/i);
    expect(RISK_DISCLAIMER).toMatch(/no protected attribute/i);
    expect(RISK_DISCLAIMER).toMatch(/residual property-scale/i);
  });
});

describe("ResidentModelCard per-head metrics (were silently invisible)", () => {
  // The backend returns `targets` as a LIST OF STRINGS, so the old
  // Object.entries(targets) yielded ["0","late"] pairs and TargetPanel got a
  // string where it expected an object -> the whole metrics section rendered
  // nothing, hiding that p_90d_12m has AUC ~0.51 (chance).
  const card: ResidentModelCardT = {
    name: "Resident Risk (multi-head)",
    targets: ["late", "arrears", "churn", "serious"],
    heads: [
      {
        name: "late_3m",
        family: "late",
        kind: "binary",
        learnable: "reliable",
        low_confidence: false,
        metrics: { auc: 0.8134, ece: 0.0871 },
      },
      {
        name: "p_90d_12m",
        family: "severity",
        kind: "binary",
        learnable: "LOW-POWER: very rare, treat as directional only",
        low_confidence: true,
        metrics: { auc: 0.5103, ece: 0.0105 },
      },
    ],
  };

  it("renders per-head metrics even though `targets` is a string list", () => {
    render(<ResidentModelCard card={card} />);
    expect(screen.getByText(/Per-head held-out metrics/)).toBeInTheDocument();
    expect(screen.getByText("late_3m")).toBeInTheDocument();
  });

  it("shows the chance-level head's AUC rather than hiding it", () => {
    render(<ResidentModelCard card={card} />);
    expect(screen.getByText(/0\.510/)).toBeInTheDocument();
  });

  it("flags low-power heads and surfaces the 'directional only' caveat", () => {
    render(<ResidentModelCard card={card} />);
    expect(screen.getByText("low power")).toBeInTheDocument();
    expect(screen.getByText(/treat as directional only/i)).toBeInTheDocument();
    expect(screen.getByText(/1 of 2 heads are low-power/i)).toBeInTheDocument();
  });

  it("does not crash on a payload with no heads at all", () => {
    render(<ResidentModelCard card={{ name: "x" }} />);
    expect(screen.queryByText(/Per-head held-out metrics/)).not.toBeInTheDocument();
  });
});

describe("LateCountBreakdownChart display math", () => {
  const pts = [
    { key: "q1" as const, label: "Q1", expected: 13.6 },
    { key: "q2" as const, label: "Q2", expected: 28.7 },
    { key: "q3" as const, label: "Q3", expected: 46.3 },
    { key: "q4" as const, label: "Q4", expected: 63.7 },
  ];

  it("renders all four quarterly checkpoints", () => {
    render(<LateCountBreakdownChart points={pts} />);
    for (const p of pts) {
      expect(screen.getByText(p.label)).toBeInTheDocument();
      expect(screen.getByText(String(p.expected))).toBeInTheDocument();
    }
  });

  it("renders whole numbers without a trailing .0", () => {
    render(<LateCountBreakdownChart points={[{ key: "q1", label: "Q1", expected: 12 }]} />);
    expect(screen.getByText("12")).toBeInTheDocument();
  });

  it("returns null for an empty series instead of dividing by zero", () => {
    const { container } = render(<LateCountBreakdownChart points={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("carries a data-bearing aria-label for screen readers", () => {
    render(<LateCountBreakdownChart points={pts} />);
    expect(screen.getByRole("img")).toHaveAttribute(
      "aria-label",
      expect.stringContaining("Q1"),
    );
  });
});
