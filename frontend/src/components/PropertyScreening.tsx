import { useEffect, useState } from "react";
import { Check, ClipboardCheck, Minus, Users, X } from "lucide-react";
import { getPropertyCandidates, listApplicants, screenApplicant } from "../api";
import type { ApplicantSummary, CandidatesResponse, ScreenResult } from "../types";

const VERDICT_TONE: Record<string, string> = {
  pass: "good",
  review: "warn",
  fail: "bad",
};

/**
 * Landlord-side screening for a listing:
 *  F1 — check a chosen applicant against THIS property's own criteria.
 *  F2 — rank all stored applicants for the vacancy (inverse of /recommend).
 * Both deterministic; no LLM.
 */
export function PropertyScreening({ propertyId }: { propertyId: string }) {
  const [applicants, setApplicants] = useState<ApplicantSummary[]>([]);
  const [sel, setSel] = useState("");
  const [screen, setScreen] = useState<ScreenResult | null>(null);
  const [screenBusy, setScreenBusy] = useState(false);
  const [cands, setCands] = useState<CandidatesResponse | null>(null);

  useEffect(() => {
    let alive = true;
    listApplicants()
      .then((a) => {
        if (!alive) return;
        setApplicants(a);
        if (a[0]) setSel(a[0].id);
      })
      .catch(() => {});
    getPropertyCandidates(propertyId, 5)
      .then((c) => alive && setCands(c))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [propertyId]);

  async function doScreen() {
    if (!sel) return;
    setScreenBusy(true);
    setScreen(null);
    try {
      setScreen(await screenApplicant(propertyId, sel));
    } catch {
      /* ignore */
    } finally {
      setScreenBusy(false);
    }
  }

  return (
    <>
      <div className="card">
        <h2 className="icon-line">
          <ClipboardCheck size={16} /> Will an applicant qualify here?
        </h2>
        <p className="muted">
          Checks a stored applicant against this listing's own criteria (income
          multiple, credit, occupancy, pets). Deterministic.
        </p>
        {applicants.length === 0 ? (
          <p className="muted">No applicants yet — add one in Apply or upload a PDF.</p>
        ) : (
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <select
              className="field"
              value={sel}
              onChange={(e) => setSel(e.target.value)}
              style={{ maxWidth: 260 }}
              aria-label="Choose applicant to screen"
            >
              {applicants.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name}
                </option>
              ))}
            </select>
            <button className="btn-small" disabled={screenBusy} onClick={doScreen}>
              {screenBusy ? "Screening…" : "Screen"}
            </button>
          </div>
        )}
        {screen && (
          <div style={{ marginTop: 12 }}>
            <span className={`badge tone-${VERDICT_TONE[screen.verdict] ?? "info"}`}>
              {screen.verdict === "pass"
                ? "Qualifies for this unit"
                : screen.verdict === "review"
                  ? "Needs review"
                  : "Does not qualify"}
            </span>
            <div style={{ marginTop: 8 }}>
              {screen.checks.map((c) => (
                <div className="screen-check" key={c.label}>
                  <span>
                    {c.label}
                    <span className="req"> · needs {c.required}</span>
                  </span>
                  <span style={{ fontVariantNumeric: "tabular-nums" }}>{c.actual}</span>
                  <span>
                    {c.ok === true ? (
                      <Check size={15} color="var(--good-text)" />
                    ) : c.ok === false ? (
                      <X size={15} color="var(--bad-text)" />
                    ) : (
                      <Minus size={15} color="var(--muted)" />
                    )}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {cands && cands.candidates.length > 0 && (
        <div className="card">
          <h2 className="icon-line">
            <Users size={16} /> Top candidates for this vacancy
          </h2>
          <p className="muted">
            Your applicant pipeline ranked for this unit ({cands.total} screened) — deterministic score.
          </p>
          {cands.candidates.map((c) => (
            <div className="cand-row" key={c.applicant_id}>
              <span
                className={`badge tone-${c.screen_passes ? "good" : c.screen_verdict === "review" ? "warn" : "bad"}`}
                title={`Screen: ${c.screen_verdict}`}
              >
                {c.screen_passes ? "qualifies" : c.screen_verdict}
              </span>
              <span>{c.name}</span>
              <span className="cand-score">{Math.round(c.score * 100)}% match</span>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
