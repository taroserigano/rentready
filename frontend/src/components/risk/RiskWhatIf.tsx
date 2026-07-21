import { useEffect, useRef, useState } from "react";
import { RotateCcw, SlidersHorizontal } from "lucide-react";
import { getApplicant, getRisk, scoreRisk } from "../../api";
import type { ApplicantProfile, RiskResult } from "../../types";
import { RiskGauge } from "./RiskGauge";
import { ReasonCodes } from "./ReasonCodes";
import { BAND_LABEL } from "./riskTone";

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

/**
 * Risk What-if — an exploratory re-score of the SELECTED applicant. Seeds an
 * editable copy of the saved profile, then debounces slider changes into
 * `scoreRisk` and shows the live gauge + drivers with a delta against the
 * applicant's saved score. Nothing here mutates the stored applicant.
 */
export function RiskWhatIf({ applicantId }: { applicantId: string }) {
  const [open, setOpen] = useState(false);
  const [base, setBase] = useState<ApplicantProfile | null>(null);
  const [profile, setProfile] = useState<ApplicantProfile | null>(null);
  const [baseline, setBaseline] = useState<number | null>(null);
  const [result, setResult] = useState<RiskResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const runId = useRef(0);

  // Seed the editable copy + baseline probability once per applicant.
  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError("");
    setResult(null);
    setBaseline(null);
    getApplicant(applicantId)
      .then((r) => {
        if (!alive) return;
        setBase(r.profile);
        setProfile(r.profile);
      })
      .catch((e) =>
        alive && setError(e instanceof Error ? e.message : String(e)),
      )
      .finally(() => alive && setLoading(false));
    // Baseline is best-effort; the delta simply hides if it never arrives.
    getRisk(applicantId)
      .then((r) => alive && setBaseline(r.probability))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [applicantId]);

  // Debounced re-score of the edited profile. A run-id guard keeps a slow
  // earlier response from clobbering a newer one.
  useEffect(() => {
    if (!open || !profile) return;
    const id = ++runId.current;
    setBusy(true);
    const t = setTimeout(() => {
      scoreRisk(profile)
        .then((res) => {
          if (id === runId.current) setResult(res);
        })
        .catch(() => {
          /* keep the last good result */
        })
        .finally(() => {
          if (id === runId.current) setBusy(false);
        });
    }, 250);
    return () => clearTimeout(t);
  }, [open, profile]);

  function patch(key: keyof ApplicantProfile, value: number) {
    setProfile((p) => (p ? { ...p, [key]: value } : p));
  }

  function reset() {
    setProfile(base);
  }

  const income = Math.round(profile?.monthly_income ?? 0);
  const rent = Math.round(profile?.desired_rent ?? 0);
  const credit = profile?.credit_score ?? 660;
  const debt = Math.round(profile?.monthly_debt_payments ?? 0);

  const baseIncome = Math.round(base?.monthly_income ?? 0);
  const baseRent = Math.round(base?.desired_rent ?? 0);
  const baseCredit = base?.credit_score ?? 660;
  const baseDebt = Math.round(base?.monthly_debt_payments ?? 0);

  const dirty =
    !!base &&
    (income !== baseIncome ||
      rent !== baseRent ||
      credit !== baseCredit ||
      debt !== baseDebt);

  // Delta vs the applicant's saved score, in probability points.
  const deltaPts =
    result != null && baseline != null
      ? Math.round(result.probability * 100) - Math.round(baseline * 100)
      : null;

  return (
    <div className="card">
      <div className="rec-head">
        <h2 style={{ margin: 0 }}>Risk what-if</h2>
        <button
          className="btn-small btn-ghost icon-line"
          style={{ marginLeft: "auto" }}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          <SlidersHorizontal size={14} />
          {open ? "Hide" : "Explore factors"}
        </button>
      </div>

      {open && (
        <>
          <p className="muted" style={{ marginTop: 4 }}>
            Drag to see how income, rent, credit, and debt would move the
            estimate. This is an exploratory estimate — it never changes the
            real applicant or their saved score.
          </p>

          {loading && (
            <p className="chat-typing" aria-label="Loading applicant" style={{ marginTop: 12 }}>
              <span />
              <span />
              <span />
            </p>
          )}

          {!loading && error && <div className="error">{error}</div>}

          {!loading && !error && profile && (
            <>
              <div className="form-grid" style={{ marginTop: 8 }}>
                <SliderRow
                  label="Monthly income"
                  min={0}
                  max={20000}
                  step={100}
                  value={income}
                  base={baseIncome}
                  format={money}
                  onChange={(n) => patch("monthly_income", n)}
                />
                <SliderRow
                  label="Desired rent"
                  min={0}
                  max={6000}
                  step={50}
                  value={rent}
                  base={baseRent}
                  format={money}
                  onChange={(n) => patch("desired_rent", n)}
                />
                <SliderRow
                  label="Credit score"
                  min={300}
                  max={850}
                  step={5}
                  value={credit}
                  base={baseCredit}
                  format={(n) => String(n)}
                  onChange={(n) => patch("credit_score", n)}
                />
                <SliderRow
                  label="Monthly debt payments"
                  min={0}
                  max={5000}
                  step={50}
                  value={debt}
                  base={baseDebt}
                  format={money}
                  onChange={(n) => patch("monthly_debt_payments", n)}
                />
              </div>

              <div
                aria-live="polite"
                style={{
                  display: "grid",
                  gridTemplateColumns: "minmax(150px, 200px) 1fr",
                  gap: 18,
                  alignItems: "center",
                  marginTop: 14,
                }}
              >
                <div>
                  {result ? (
                    <>
                      <RiskGauge
                        probability={result.probability}
                        band={result.band}
                      />
                      <div
                        style={{
                          textAlign: "center",
                          fontSize: 12,
                          color: "var(--text-2)",
                          marginTop: 4,
                        }}
                      >
                        {BAND_LABEL[result.band]} risk{busy && " · …"}
                      </div>
                    </>
                  ) : (
                    <p className="muted" style={{ margin: 0, fontSize: 13 }}>
                      Estimating…
                    </p>
                  )}
                </div>
                <div>
                  <div className="eyebrow">Key factors</div>
                  {result ? (
                    <ReasonCodes codes={result.reason_codes} />
                  ) : (
                    <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
                      Adjust a factor to see the drivers.
                    </p>
                  )}
                </div>
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
                {deltaPts != null && baseline != null && (
                  <span
                    className={`delta ${
                      deltaPts > 0 ? "up" : deltaPts < 0 ? "down" : "zero"
                    }`}
                  >
                    {deltaPts > 0 ? "↑ " : deltaPts < 0 ? "↓ " : "→ "}
                    {Math.abs(deltaPts)} pts from {Math.round(baseline * 100)}%
                    {deltaPts === 0 && " (no change)"}
                  </span>
                )}
                {dirty && (
                  <>
                    <span className="badge tone-warn">
                      Exploratory — not the saved score
                    </span>
                    <button
                      className="btn-small btn-ghost icon-line"
                      onClick={reset}
                    >
                      <RotateCcw size={13} /> Reset
                    </button>
                  </>
                )}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
