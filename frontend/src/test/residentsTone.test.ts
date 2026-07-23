import { describe, expect, it } from "vitest";
import {
  BAND_LABEL,
  BAND_TONE,
  FAMILY_HINT,
  FAMILY_LABEL,
  HEAD_LABEL,
  RESIDENT_DISCLAIMER,
  STATUS_ORDER,
  STATUS_STYLE,
  bandLabel,
  churnLabel,
  churnTone,
  formatProbRange,
  gradeTone,
  headLabel,
  pct,
  usd,
} from "../components/residents/residentsTone";
import type { HealthGrade, LedgerStatus, RiskBand } from "../types";

describe("residentsTone: band re-exports", () => {
  it("re-exports BAND_LABEL / BAND_TONE from riskTone", () => {
    expect(BAND_LABEL).toEqual({ low: "Low", medium: "Moderate", high: "Elevated" });
    expect(BAND_TONE).toEqual({ low: "good", medium: "warn", high: "bad" });
  });

  it("bandLabel maps every band through BAND_LABEL", () => {
    (["low", "medium", "high"] as RiskBand[]).forEach((b) => {
      expect(bandLabel(b)).toBe(BAND_LABEL[b]);
    });
  });
});

describe("residentsTone: churnLabel / churnTone", () => {
  it("labels each real band and the not_applicable sentinel", () => {
    expect(churnLabel("low")).toBe("Low");
    expect(churnLabel("medium")).toBe("Moderate");
    expect(churnLabel("high")).toBe("Elevated");
    expect(churnLabel("not_applicable")).toBe("Not applicable");
  });

  it("tones each real band good/warn/bad, not_applicable => info", () => {
    expect(churnTone("low")).toBe("good");
    expect(churnTone("medium")).toBe("warn");
    expect(churnTone("high")).toBe("bad");
    expect(churnTone("not_applicable")).toBe("info");
  });
});

describe("residentsTone: RESIDENT_DISCLAIMER", () => {
  it("frames as decision-support only and lists protected classes", () => {
    expect(RESIDENT_DISCLAIMER).toMatch(/Decision-support only/);
    expect(RESIDENT_DISCLAIMER).toMatch(/synthetic data/);
    expect(RESIDENT_DISCLAIMER).toMatch(/human\s*\n?\s*reviewer|human reviewer/);
    expect(RESIDENT_DISCLAIMER).toMatch(/race, national origin, sex/);
  });
});

describe("residentsTone: STATUS_STYLE / STATUS_ORDER", () => {
  it("has a style for every ledger status with matching cls + tone", () => {
    const expected: Record<LedgerStatus, { tone: string; cls: string; label: string }> = {
      paid: { tone: "good", cls: "is-paid", label: "Paid on time" },
      paid_late: { tone: "warn", cls: "is-late", label: "Paid late" },
      partial: { tone: "info", cls: "is-partial", label: "Partial" },
      missed: { tone: "bad", cls: "is-missed", label: "Missed" },
    };
    (Object.keys(expected) as LedgerStatus[]).forEach((s) => {
      expect(STATUS_STYLE[s].tone).toBe(expected[s].tone);
      expect(STATUS_STYLE[s].cls).toBe(expected[s].cls);
      expect(STATUS_STYLE[s].label).toBe(expected[s].label);
      expect(STATUS_STYLE[s].token).toMatch(/^var\(--chart-\d\)$/);
    });
  });

  it("STATUS_ORDER lists all four statuses in legend order", () => {
    expect(STATUS_ORDER).toEqual(["paid", "paid_late", "partial", "missed"]);
  });
});

describe("residentsTone: usd", () => {
  it("formats whole-dollar currency with no cents", () => {
    expect(usd(1240)).toBe("$1,240");
    expect(usd(0)).toBe("$0");
    expect(usd(1234567)).toBe("$1,234,567");
  });

  it("rounds to whole dollars", () => {
    expect(usd(1240.75)).toBe("$1,241");
  });

  it("returns an em dash for null / undefined / NaN", () => {
    expect(usd(null)).toBe("—");
    expect(usd(undefined)).toBe("—");
    expect(usd(NaN)).toBe("—");
  });

  it("handles negatives", () => {
    expect(usd(-500)).toBe("-$500");
  });
});

describe("residentsTone: pct", () => {
  it("renders a rounded whole percent from a 0..1 probability", () => {
    expect(pct(0.42)).toBe("42%");
    expect(pct(0)).toBe("0%");
    expect(pct(1)).toBe("100%");
    expect(pct(0.005)).toBe("1%");
  });

  it("returns an em dash for null / undefined / NaN", () => {
    expect(pct(null)).toBe("—");
    expect(pct(undefined)).toBe("—");
    expect(pct(NaN)).toBe("—");
  });
});

describe("residentsTone: gradeTone", () => {
  it("A and B read good", () => {
    expect(gradeTone("A")).toBe("good");
    expect(gradeTone("B")).toBe("good");
  });
  it("C warns", () => {
    expect(gradeTone("C")).toBe("warn");
  });
  it("D and F read bad", () => {
    expect(gradeTone("D")).toBe("bad");
    expect(gradeTone("F")).toBe("bad");
  });
  it("unknown grades fall back to info", () => {
    expect(gradeTone("Z" as HealthGrade)).toBe("info");
    expect(gradeTone("")).toBe("info");
  });
});

describe("residentsTone: FAMILY_LABEL / FAMILY_HINT", () => {
  it("labels all six known families", () => {
    ["late", "frequency", "severity", "arrears", "cure", "retention"].forEach((f) => {
      expect(typeof FAMILY_LABEL[f]).toBe("string");
      expect(FAMILY_LABEL[f].length).toBeGreaterThan(0);
      expect(typeof FAMILY_HINT[f]).toBe("string");
    });
    expect(FAMILY_LABEL.late).toBe("Late-payment outlook");
    expect(FAMILY_LABEL.arrears).toBe("Arrears ($)");
  });
});

describe("residentsTone: headLabel", () => {
  it("returns the mapped label for a known head", () => {
    expect(headLabel("late_3m")).toBe("Next quarter");
    expect(headLabel("serious")).toBe("Serious delinquency");
    expect(HEAD_LABEL.churn).toBe("Non-renewal (≤6mo)");
  });

  it("de-slugs an unknown head name", () => {
    expect(headLabel("some_new_head")).toBe("some new head");
    expect(headLabel("plain")).toBe("plain");
  });
});

describe("residentsTone: formatProbRange (re-export)", () => {
  it("renders ~p% (lo–hi%)", () => {
    expect(formatProbRange(0.2, [0.12, 0.28])).toBe("~20% (12–28%)");
  });
});
