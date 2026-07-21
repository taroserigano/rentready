import { useEffect, useMemo, useRef, useState } from "react";
import { RotateCcw, SlidersHorizontal, Target } from "lucide-react";
import type { ApplicantProfile, GoalSeekResult, SimulateResponse } from "../types";
import { goalSeek, simulate } from "../api";

const VERDICT_TONE: Record<string, string> = {
  qualified: "good",
  needs_review: "warn",
  not_qualified: "bad",
};

function money(n: number): string {
  return `$${Math.round(n).toLocaleString()}`;
}

function SliderRow({
  label,
  min,
  max,
  step,
  value,
  base,
  format,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  base: number;
  format: (n: number) => string;
  onChange: (n: number) => void;
}) {
  const delta = value - base;
  const deltaClass = delta > 0 ? "up" : delta < 0 ? "down" : "zero";
  return (
    <label className="field" style={{ gridColumn: "1 / -1" }}>
      <span style={{ display: "flex", justifyContent: "space-between" }}>
        <span>{label}</span>
        <span style={{ fontVariantNumeric: "tabular-nums", color: "var(--text)" }}>
          {format(value)}{" "}
          {delta !== 0 && (
            <span className={`delta ${deltaClass}`} style={{ marginLeft: 4 }}>
              ({delta > 0 ? "+" : ""}
              {format(delta)})
            </span>
          )}
        </span>
      </span>
      <input
        className="slider"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        aria-valuetext={format(value)}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export function WhatIfSimulator({
  profile,
  applicantId,
}: {
  profile: ApplicantProfile;
  applicantId: string;
}) {
  const [open, setOpen] = useState(false);
  const baseIncome = Math.round(profile.monthly_income) || 0;
  const baseRent = Math.round(profile.desired_rent) || 0;
  const baseCredit = profile.credit_score ?? 660;

  const [income, setIncome] = useState(baseIncome);
  const [rent, setRent] = useState(baseRent);
  const [credit, setCredit] = useState(baseCredit);
  const [sim, setSim] = useState<SimulateResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const runId = useRef(0);
  const [goal, setGoal] = useState<GoalSeekResult | null>(null);
  const [goalBusy, setGoalBusy] = useState(false);

  async function runGoal(solveFor: "monthly_income" | "desired_rent") {
    setGoalBusy(true);
    setGoal(null);
    try {
      setGoal(await goalSeek(applicantId, solveFor));
    } catch {
      /* ignore */
    } finally {
      setGoalBusy(false);
    }
  }

  const dirty =
    income !== baseIncome || rent !== baseRent || credit !== baseCredit;
  const ratio = rent > 0 ? income / rent : 0;

  // Debounced server re-run (deterministic; skips the LLM). A run-id guard
  // stops a slow earlier response from overwriting a newer one.
  useEffect(() => {
    if (!open) return;
    const id = ++runId.current;
    setBusy(true);
    const t = setTimeout(() => {
      simulate(applicantId, {
        monthly_income: income,
        desired_rent: rent,
        credit_score: credit,
      })
        .then((res) => {
          if (id === runId.current) setSim(res);
        })
        .catch(() => {
          /* keep last good result; client ratio still updates */
        })
        .finally(() => {
          if (id === runId.current) setBusy(false);
        });
    }, 300);
    return () => clearTimeout(t);
  }, [open, income, rent, credit, applicantId]);

  function reset() {
    setIncome(baseIncome);
    setRent(baseRent);
    setCredit(baseCredit);
  }

  const topMatches = useMemo(
    () => sim?.recommendations.recommendations.slice(0, 3) ?? [],
    [sim],
  );
  const verdict = sim?.eligibility.verdict;

  return (
    <div className="card">
      <div className="rec-head">
        <h2 style={{ margin: 0 }}>3b. What-if</h2>
        <button
          className="btn-small btn-ghost icon-line"
          style={{ marginLeft: "auto" }}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          <SlidersHorizontal size={14} />
          {open ? "Hide" : "Adjust income & rent"}
        </button>
      </div>

      {open && (
        <>
          <p className="muted" style={{ marginTop: 4 }}>
            Drag to see eligibility and matches recompute — deterministic, and it
            works with no LLM. Nothing here changes the real applicant.
          </p>

          <div className="form-grid" style={{ marginTop: 8 }}>
            <SliderRow
              label="Monthly income"
              min={0}
              max={20000}
              step={100}
              value={income}
              base={baseIncome}
              format={money}
              onChange={setIncome}
            />
            <SliderRow
              label="Desired rent"
              min={0}
              max={6000}
              step={50}
              value={rent}
              base={baseRent}
              format={money}
              onChange={setRent}
            />
            <SliderRow
              label="Credit score"
              min={300}
              max={850}
              step={5}
              value={credit}
              base={baseCredit}
              format={(n) => String(n)}
              onChange={setCredit}
            />
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              flexWrap: "wrap",
              marginTop: 12,
            }}
          >
            <span className="badge tone-info">
              Income is {ratio.toFixed(1)}× rent
            </span>
            {verdict && (
              <span className={`badge tone-${VERDICT_TONE[verdict] ?? "info"}`}>
                {verdict.replace(/_/g, " ")}
                {busy && " · …"}
              </span>
            )}
            {dirty && (
              <>
                <span className="badge tone-warn">Simulated — not the real figures</span>
                <button className="btn-small btn-ghost icon-line" onClick={reset}>
                  <RotateCcw size={13} /> Reset to actual
                </button>
              </>
            )}
          </div>

          <div style={{ marginTop: 14, paddingTop: 12, borderTop: "1px solid var(--line-soft)" }}>
            <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
              Goal-seek — solve for the exact threshold to reach “qualified”:
            </div>
            <div className="chip-row" style={{ marginTop: 0 }}>
              <button className="chip" disabled={goalBusy} onClick={() => runGoal("monthly_income")}>
                <Target size={13} /> Min income to qualify
              </button>
              <button className="chip" disabled={goalBusy} onClick={() => runGoal("desired_rent")}>
                <Target size={13} /> Max rent to qualify
              </button>
            </div>
            {goal && (
              <div style={{ marginTop: 8, fontSize: 13 }}>
                {goal.achievable ? (
                  <span className="icon-line">
                    <span className="badge tone-good">
                      {goal.solve_for === "monthly_income"
                        ? `Income ≥ ${money(goal.threshold!)}/mo`
                        : `Rent ≤ ${money(goal.threshold!)}/mo`}
                    </span>
                    <button
                      className="btn-small btn-ghost"
                      onClick={() =>
                        goal.solve_for === "monthly_income"
                          ? setIncome(goal.threshold!)
                          : setRent(goal.threshold!)
                      }
                    >
                      Apply
                    </button>
                  </span>
                ) : (
                  <span className="badge tone-warn">{goal.reason}</span>
                )}
              </div>
            )}
          </div>

          {topMatches.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
                Top matches at these figures:
              </div>
              {topMatches.map((r) => (
                <div
                  key={r.property_id}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "4px 0",
                    borderBottom: "1px solid var(--line-soft)",
                    fontSize: 13,
                  }}
                >
                  <span>
                    {r.name}{" "}
                    <span className="muted">· {Math.round(r.score * 100)}% match</span>
                  </span>
                  <span style={{ fontVariantNumeric: "tabular-nums" }}>
                    {money(r.monthly_rent)}/mo
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
