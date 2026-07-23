import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  askResidentChat,
  askRiskChat,
  applyForm,
  getHealth,
  getResident,
  getResidentsHealth,
  getRisk,
  listPropertyResidents,
  listResidents,
  listRisk,
  scoreResident,
  scoreRisk,
  simulate,
  evalReportUrl,
  pdfUrl,
  leasePdfUrl,
  toursIcsUrl,
} from "../api";
import type { ApplicantProfile } from "../types";

const BASE = "http://localhost:8000";

/** A minimal ok Response stub whose json() resolves to `body`. */
function okJson(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response;
}

/** A non-ok Response stub with a text body (JSON string or raw). */
function errRes(status: number, text: string): Response {
  return {
    ok: false,
    status,
    json: async () => JSON.parse(text),
    text: async () => text,
  } as unknown as Response;
}

let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("api: static URL helpers", () => {
  it("build the expected absolute URLs", () => {
    expect(evalReportUrl).toBe(`${BASE}/evals/report`);
    expect(pdfUrl("APP-1")).toBe(`${BASE}/applicants/APP-1/pdf`);
    expect(leasePdfUrl("PROP 4")).toBe(`${BASE}/concierge/lease/PROP%204/pdf`);
    expect(toursIcsUrl("BK-9")).toBe(`${BASE}/tours/BK-9/calendar.ics`);
  });
});

describe("api: GET wrappers", () => {
  it("getHealth hits /health and returns parsed JSON", async () => {
    fetchMock.mockResolvedValue(okJson({ status: "ok", anthropic: true }));
    const out = await getHealth();
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/health`);
    expect(out).toEqual({ status: "ok", anthropic: true });
  });

  it("listRisk hits /risk", async () => {
    const body = { rows: [], total: 0, scored: 0, avg_probability: 0, high_risk_pct: 0 };
    fetchMock.mockResolvedValue(okJson(body));
    const out = await listRisk();
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/risk`);
    expect(out).toEqual(body);
  });

  it("getRisk URL-encodes the applicant id", async () => {
    fetchMock.mockResolvedValue(okJson({ applicant_id: "A/B" }));
    await getRisk("A/B");
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/risk/A%2FB`);
  });

  it("getResident URL-encodes the resident id", async () => {
    fetchMock.mockResolvedValue(okJson({ resident: {}, predictions: {} }));
    await getResident("R 1");
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/residents/R%201`);
  });

  it("listResidents without a property hits the bare endpoint", async () => {
    fetchMock.mockResolvedValue(okJson({ residents: [], count: 0, source: "x" }));
    await listResidents();
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/residents`);
  });

  it("listResidents scopes to a property via query string", async () => {
    fetchMock.mockResolvedValue(okJson({ residents: [], count: 0, source: "x" }));
    await listResidents("PROP 7");
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/residents?property_id=PROP%207`);
  });

  it("listPropertyResidents encodes the property id", async () => {
    fetchMock.mockResolvedValue(okJson({ residents: [], rollup: { resident_count: 0 } }));
    await listPropertyResidents("PROP/9");
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/properties/PROP%2F9/residents`);
  });
});

describe("api: getResidentsHealth normalization", () => {
  it("returns a bare array unchanged", async () => {
    const arr = [{ property_id: "P1", name: "A", score: 90, grade: "A" }];
    fetchMock.mockResolvedValue(okJson(arr));
    const out = await getResidentsHealth();
    expect(fetchMock).toHaveBeenCalledWith(`${BASE}/residents/health`);
    expect(out).toEqual(arr);
  });

  it("unwraps {properties: [...]}", async () => {
    const arr = [{ property_id: "P1" }];
    fetchMock.mockResolvedValue(okJson({ properties: arr }));
    expect(await getResidentsHealth()).toEqual(arr);
  });

  it("unwraps {ranking: [...]} and {rows: [...]}", async () => {
    fetchMock.mockResolvedValueOnce(okJson({ ranking: [{ property_id: "R" }] }));
    expect(await getResidentsHealth()).toEqual([{ property_id: "R" }]);
    fetchMock.mockResolvedValueOnce(okJson({ rows: [{ property_id: "W" }] }));
    expect(await getResidentsHealth()).toEqual([{ property_id: "W" }]);
  });

  it("returns [] when the shape is unrecognized", async () => {
    fetchMock.mockResolvedValue(okJson({ nope: 1 }));
    expect(await getResidentsHealth()).toEqual([]);
  });
});

describe("api: POST wrappers send method + JSON body", () => {
  it("simulate posts applicant_id + overrides", async () => {
    fetchMock.mockResolvedValue(okJson({ eligibility: {}, recommendations: {} }));
    await simulate("APP-1", { monthly_income: 6000, desired_rent: 1800 });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/simulate`);
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(init.body)).toEqual({
      applicant_id: "APP-1",
      monthly_income: 6000,
      desired_rent: 1800,
    });
  });

  it("askRiskChat posts the request body to /risk/chat", async () => {
    fetchMock.mockResolvedValue(okJson({ answer: "hi", source: "rules" }));
    await askRiskChat({ question: "why?", applicant_id: "A1" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/risk/chat`);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ question: "why?", applicant_id: "A1" });
  });

  it("askResidentChat posts to /residents/chat", async () => {
    fetchMock.mockResolvedValue(okJson({ answer: "ok", source: "rules" }));
    await askResidentChat({ question: "how?", resident_id: "R1" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/residents/chat`);
    expect(JSON.parse(init.body)).toEqual({ question: "how?", resident_id: "R1" });
  });

  it("scoreResident POSTs to the score endpoint (no body)", async () => {
    fetchMock.mockResolvedValue(okJson({ late: {}, arrears: {}, churn: {}, serious: {} }));
    await scoreResident("R/2");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/residents/R%2F2/score`);
    expect(init.method).toBe("POST");
  });

  it("scoreRisk wraps the profile under {profile}", async () => {
    fetchMock.mockResolvedValue(okJson({ applicant_id: "adhoc" }));
    const profile = { name: "X", monthly_income: 5000 } as unknown as ApplicantProfile;
    await scoreRisk(profile);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/risk/score`);
    expect(JSON.parse(init.body)).toEqual({ profile });
  });

  it("applyForm posts the profile to /apply", async () => {
    fetchMock.mockResolvedValue(okJson({ applicant_id: "new" }));
    const profile = { name: "Y" } as unknown as ApplicantProfile;
    await applyForm(profile);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${BASE}/apply`);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual(profile);
  });
});

describe("api: error handling in json()", () => {
  it("surfaces a FastAPI {detail} message on non-ok", async () => {
    fetchMock.mockResolvedValue(errRes(404, JSON.stringify({ detail: "not found" })));
    await expect(getRisk("nope")).rejects.toThrow("not found");
  });

  it("falls back to `status: text` when the body is not JSON", async () => {
    fetchMock.mockResolvedValue(errRes(500, "boom"));
    await expect(getHealth()).rejects.toThrow("500: boom");
  });

  it("uses `status: text` when detail is missing/blank", async () => {
    fetchMock.mockResolvedValue(errRes(400, JSON.stringify({ detail: "" })));
    await expect(getHealth()).rejects.toThrow("400:");
  });
});
