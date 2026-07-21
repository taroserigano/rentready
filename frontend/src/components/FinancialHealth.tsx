import { Home, PiggyBank, Scale, Wallet } from "lucide-react";
import type { ApplicantProfile } from "../types";

type Tone = "good" | "warn" | "bad";

function money(n: number): string {
  return `$${Math.round(n).toLocaleString()}`;
}

/** A labeled meter row: title, value, and a tone-graded fill (clamped 0–100%). */
function MeterRow({
  icon,
  label,
  value,
  pct,
  tone,
  note,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  pct: number;
  tone: Tone;
  note: string;
}) {
  return (
    <div
      className="fh-row"
      role="meter"
      aria-label={`${label}: ${value}, ${note}`}
      aria-valuenow={Math.round(Math.max(0, Math.min(100, pct)))}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className="fh-head">
        <span className="icon-line">
          {icon} {label}
        </span>
        <span className="fh-val">
          {value} · {note}
        </span>
      </div>
      <div className="meter">
        <i className={tone} style={{ width: `${Math.max(2, Math.min(100, pct))}%` }} />
      </div>
    </div>
  );
}

/**
 * Deterministic financial-health read from the extracted/entered profile —
 * no LLM, no network. Renders only the metrics whose inputs are present.
 */
export function FinancialHealth({ profile }: { profile: ApplicantProfile }) {
  const income = profile.monthly_income + (profile.other_income_monthly ?? 0);
  const rent = profile.desired_rent;
  const debt = profile.monthly_debt_payments ?? 0;
  const savings = profile.savings_balance ?? null;

  if (!(income > 0) || !(rent > 0)) return null;

  const rentPct = (rent / income) * 100;
  const dtiPct = ((debt + rent) / income) * 100;
  const runway = savings != null && rent > 0 ? savings / rent : null;
  const maxRent = income / 3;

  const rentTone: Tone = rentPct <= 30 ? "good" : rentPct <= 40 ? "warn" : "bad";
  const dtiTone: Tone = dtiPct <= 36 ? "good" : dtiPct <= 43 ? "warn" : "bad";
  const runwayTone: Tone =
    runway == null ? "warn" : runway >= 3 ? "good" : runway >= 1 ? "warn" : "bad";

  const word = (t: Tone) =>
    t === "good" ? "healthy" : t === "warn" ? "elevated" : "high";

  return (
    <div className="card">
      <h2>3a. Financial health</h2>
      <p className="muted">
        Deterministic read of the application — recommended max rent{" "}
        <strong style={{ color: "var(--accent-text)" }}>{money(maxRent)}/mo</strong>{" "}
        at the 3× income rule.
      </p>

      <div className="fh-grid">
        <div className="stat-tile">
          <div className="label">Effective income</div>
          <div className="value">{money(income)}</div>
          <div className="sub">incl. other income</div>
        </div>
        <div className="stat-tile">
          <div className="label">Recommended max rent</div>
          <div className="value">{money(maxRent)}</div>
          <div className="sub">at 3× income</div>
        </div>
        <div className="stat-tile">
          <div className="label">Credit</div>
          <div className="value">
            {profile.credit_score != null ? profile.credit_score : "—"}
          </div>
          <div className="sub">
            {profile.credit_score == null
              ? "not provided"
              : profile.credit_score >= 720
                ? "excellent"
                : profile.credit_score >= 660
                  ? "good"
                  : profile.credit_score >= 620
                    ? "fair"
                    : "below minimum"}
          </div>
        </div>
      </div>

      <MeterRow
        icon={<Wallet size={12} />}
        label="Rent burden"
        value={`${Math.round(rentPct)}%`}
        pct={rentPct}
        tone={rentTone}
        note={word(rentTone)}
      />
      <MeterRow
        icon={<Scale size={12} />}
        label="Debt + rent to income"
        value={`${Math.round(dtiPct)}%`}
        pct={dtiPct}
        tone={dtiTone}
        note={word(dtiTone)}
      />
      {runway != null ? (
        <MeterRow
          icon={<PiggyBank size={12} />}
          label="Savings runway"
          value={`${runway.toFixed(1)} mo`}
          pct={(runway / 6) * 100}
          tone={runwayTone}
          note={runwayTone === "good" ? "healthy" : runwayTone === "warn" ? "thin" : "low"}
        />
      ) : (
        <div className="fh-row">
          <div className="fh-head">
            <span className="icon-line">
              <PiggyBank size={12} /> Savings runway
            </span>
            <span className="fh-val" style={{ color: "var(--faint)" }}>
              — · not provided
            </span>
          </div>
          <div className="meter">
            <i style={{ width: "0%" }} />
          </div>
        </div>
      )}
      <div className="icon-line muted" style={{ fontSize: 12, marginTop: 10 }}>
        <Home size={12} />
        {debt > 0
          ? `Debt payments of ${money(debt)}/mo bring total obligations to ${Math.round(dtiPct)}% of income.`
          : `No monthly debt on file — obligations are rent only (${Math.round(rentPct)}% of income).`}
      </div>
    </div>
  );
}
