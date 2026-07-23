import { describe, expect, it } from "vitest";
import {
  BAND_COLOR,
  BAND_LABEL,
  BAND_METER,
  BAND_TONE,
  RISK_DISCLAIMER,
  formatProbRange,
} from "../components/risk/riskTone";
import type { RiskBand } from "../types";

const BANDS: RiskBand[] = ["low", "medium", "high"];

describe("riskTone: BAND_LABEL", () => {
  it("maps each band to its UI label", () => {
    expect(BAND_LABEL.low).toBe("Low");
    expect(BAND_LABEL.medium).toBe("Moderate");
    expect(BAND_LABEL.high).toBe("Elevated");
  });
  it("labels every band non-empty", () => {
    BANDS.forEach((b) => expect(BAND_LABEL[b].length).toBeGreaterThan(0));
  });
});

describe("riskTone: BAND_TONE / BAND_METER", () => {
  it("tones low=good, medium=warn, high=bad", () => {
    expect(BAND_TONE).toEqual({ low: "good", medium: "warn", high: "bad" });
  });
  it("BAND_METER matches BAND_TONE for every band", () => {
    BANDS.forEach((b) => expect(BAND_METER[b]).toBe(BAND_TONE[b]));
  });
});

describe("riskTone: BAND_COLOR", () => {
  it("maps every band to a chart CSS var token", () => {
    expect(BAND_COLOR.low).toBe("var(--chart-2)");
    expect(BAND_COLOR.medium).toBe("var(--chart-3)");
    expect(BAND_COLOR.high).toBe("var(--chart-4)");
    BANDS.forEach((b) => expect(BAND_COLOR[b]).toMatch(/^var\(--chart-\d\)$/));
  });
});

describe("riskTone: RISK_DISCLAIMER", () => {
  it("frames decision-support and protected classes", () => {
    expect(RISK_DISCLAIMER).toMatch(/Decision-support only/);
    expect(RISK_DISCLAIMER).toMatch(/synthetic data/);
    expect(RISK_DISCLAIMER).toMatch(/race, national origin, sex/);
  });
});

describe("riskTone: formatProbRange", () => {
  it("rounds probability and range to whole percents", () => {
    expect(formatProbRange(0.2, [0.12, 0.28])).toBe("~20% (12–28%)");
  });
  it("handles 0 and 1 bounds", () => {
    expect(formatProbRange(0, [0, 0])).toBe("~0% (0–0%)");
    expect(formatProbRange(1, [0.9, 1])).toBe("~100% (90–100%)");
  });
  it("rounds .5 up", () => {
    expect(formatProbRange(0.005, [0.004, 0.015])).toBe("~1% (0–2%)");
  });
});
