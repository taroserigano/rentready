import { useEffect, useMemo, useState } from "react";
import { getResident } from "../../api";
import type { LedgerEntry, ResidentDetail as ResidentDetailT } from "../../types";
import { PaymentHistoryTimeline } from "./PaymentHistoryTimeline";
import {
  ArrearsPredictionCard,
  ChurnPredictionCard,
  LatePredictionCard,
  SeriousPredictionCard,
} from "./PredictionCards";
import { STATUS_STYLE, usd } from "./residentsTone";

const LEDGER_PREVIEW = 12;

function errText(e: unknown): string {
  if (e instanceof TypeError) return "Could not reach the server. Is the backend running?";
  if (e instanceof Error && e.message) return e.message;
  return "Could not load this resident.";
}

/** Small derived-stats strip computed straight from the ledger (leakage-free). */
function derived(ledger: LedgerEntry[]) {
  if (ledger.length === 0) {
    return { balance: 0, onTimeRate: 0, lateFees: 0, notices: 0, months: 0 };
  }
  const balance = ledger[ledger.length - 1]?.balance_after ?? 0;
  const last12 = ledger.slice(-12);
  const onTime = last12.filter((e) => e.on_time).length;
  const lateFees = last12.reduce((a, e) => a + (e.late_fee || 0), 0);
  const notices = last12.reduce((a, e) => a + (e.notices_sent || 0), 0);
  return {
    balance,
    onTimeRate: last12.length ? onTime / last12.length : 0,
    lateFees,
    notices,
    months: ledger.length,
  };
}

/**
 * Full resident detail: the four predictions as cards (late & serious via the
 * shared RiskGauge + ReasonCodes, arrears as a $ card, churn as a gauge or a
 * "not applicable" note), a 5-year payment-history heatmap + balance sparkline,
 * and the rent ledger.
 */
export function ResidentDetail({
  residentId,
}: {
  residentId: string;
}) {
  const [data, setData] = useState<ResidentDetailT | null>(null);
  const [error, setError] = useState("");
  const [showAllLedger, setShowAllLedger] = useState(false);

  useEffect(() => {
    let alive = true;
    setData(null);
    setError("");
    setShowAllLedger(false);
    getResident(residentId)
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(errText(e)));
    return () => {
      alive = false;
    };
  }, [residentId]);

  const ledger = data?.resident.ledger ?? [];
  const stats = useMemo(() => derived(ledger), [ledger]);
  const ledgerRows = useMemo(
    () => [...ledger].reverse().slice(0, showAllLedger ? ledger.length : LEDGER_PREVIEW),
    [ledger, showAllLedger],
  );

  if (error) {
    return (
      <div className="card">
        <div className="error">{error}</div>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="card">
        <p className="muted" style={{ margin: 0 }}>Scoring resident…</p>
      </div>
    );
  }

  const r = data.resident;
  const p = data.predictions;

  return (
    <div style={{ display: "grid", gap: 18 }}>
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, flexWrap: "wrap" }}>
          <div>
            <h2 style={{ margin: "0 0 2px" }}>Unit {r.unit_id}</h2>
            <p className="muted" style={{ margin: 0, fontSize: 13 }}>
              {r.resident_id} · {r.property_id} · base rent {usd(r.base_rent)}
              {r.autopay_enrolled ? " · autopay on" : ""}
            </p>
          </div>
        </div>

        <div className="stat-grid" style={{ marginTop: 14 }}>
          <div className="stat-tile">
            <div className="label">Tenure</div>
            <div className="value">{r.tenure_months ?? stats.months}<span className="muted" style={{ fontSize: 15, fontWeight: 500 }}> mo</span></div>
          </div>
          <div className="stat-tile">
            <div className="label">Current balance</div>
            <div className={`value ${(r.current_balance ?? stats.balance) > 0 ? "bad" : "good"}`}>
              {usd(r.current_balance ?? stats.balance)}
            </div>
          </div>
          <div className="stat-tile">
            <div className="label">On-time (12mo)</div>
            <div className="value">{Math.round(stats.onTimeRate * 100)}%</div>
          </div>
          <div className="stat-tile">
            <div className="label">Late fees (12mo)</div>
            <div className="value">{usd(stats.lateFees)}</div>
          </div>
          <div className="stat-tile">
            <div className="label">Notices (12mo)</div>
            <div className="value">{stats.notices}</div>
          </div>
        </div>
      </div>

      {/* Four predictions */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: 18 }}>
        <LatePredictionCard pred={p.late} />
        <SeriousPredictionCard pred={p.serious} />
        <ArrearsPredictionCard pred={p.arrears} />
        <ChurnPredictionCard pred={p.churn} />
      </div>

      {/* Payment history */}
      <div className="card" style={{ marginTop: 0 }}>
        <h2>Payment history — 5 years</h2>
        <p className="muted" style={{ margin: "0 0 12px", fontSize: 13 }}>
          One cell per month, oldest to newest. Hover any cell for the month, status, days late, and amount.
        </p>
        <PaymentHistoryTimeline ledger={ledger} />
      </div>

      {/* Ledger table */}
      <div className="card" style={{ marginTop: 0 }}>
        <h2>Rent ledger</h2>
        <div className="table-scroll" style={{ marginTop: 8 }}>
          <table className="table">
            <thead>
              <tr>
                <th>Month</th>
                <th>Charged</th>
                <th>Paid</th>
                <th>Days late</th>
                <th>Late fee</th>
                <th>Status</th>
                <th>Balance</th>
                <th>Notices</th>
              </tr>
            </thead>
            <tbody>
              {ledgerRows.map((e) => {
                const s = STATUS_STYLE[e.status];
                return (
                  <tr key={e.period}>
                    <td>{e.period}</td>
                    <td>{usd(e.rent_charged)}</td>
                    <td>{usd(e.amount_paid)}</td>
                    <td>{e.days_late || "—"}</td>
                    <td>{e.late_fee ? usd(e.late_fee) : "—"}</td>
                    <td>
                      <span className={`badge tone-${s.tone}`}>{s.label}</span>
                    </td>
                    <td>{usd(e.balance_after)}</td>
                    <td className="secondary">
                      {e.notices_sent}
                      {e.notices_sent > 0 ? ` (${e.notice_responded} ans)` : ""}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {ledger.length > LEDGER_PREVIEW && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10 }}>
            <span className="muted" style={{ fontSize: 12 }}>
              Showing {ledgerRows.length} of {ledger.length} months
            </span>
            <button type="button" className="btn-small btn-ghost" onClick={() => setShowAllLedger((v) => !v)}>
              {showAllLedger ? "Show recent 12" : "Show all 60"}
            </button>
          </div>
        )}
      </div>

    </div>
  );
}
