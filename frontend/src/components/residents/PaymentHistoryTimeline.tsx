import { useMemo } from "react";
import type { LedgerEntry } from "../../types";
import { STATUS_ORDER, STATUS_STYLE, usd } from "./residentsTone";

const MONTHS = 60;

/** "2026-07" → "Jul 2026" for tooltips. */
function periodLabel(period: string): string {
  const [y, m] = period.split("-").map(Number);
  if (!y || !m) return period;
  const d = new Date(Date.UTC(y, m - 1, 1));
  return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
}

/**
 * Five-year (60-month) rent-payment heatmap. One cell per month, coloured by
 * status through TOKEN classes only (see `.pay-cell.is-*` in index.css) — never
 * a hardcoded colour. Every cell also carries a text tooltip (month, status,
 * days late, amount) and an aria-label, so the reading never relies on colour.
 *
 * A small balance sparkline sits underneath, tracing the rolling balance owed;
 * its stroke is a CSS token too.
 */
export function PaymentHistoryTimeline({ ledger }: { ledger: LedgerEntry[] }) {
  // Newest-last; pad the front with blanks so the grid always reads as 5 years
  // (short-tenure residents have < 60 entries).
  const cells = useMemo(() => {
    const recent = ledger.slice(-MONTHS);
    const pad = Math.max(0, MONTHS - recent.length);
    return [
      ...Array.from({ length: pad }, () => null),
      ...recent,
    ] as (LedgerEntry | null)[];
  }, [ledger]);

  const spark = useMemo(() => buildSparkline(ledger.slice(-MONTHS)), [ledger]);

  return (
    <div>
      <div className="pay-grid" role="img" aria-label="Five-year rent payment history heatmap">
        {cells.map((e, i) => {
          if (!e) {
            return <span key={`pad-${i}`} className="pay-cell is-empty" aria-hidden />;
          }
          const s = STATUS_STYLE[e.status];
          const late = e.days_late > 0 ? `, ${e.days_late} day${e.days_late === 1 ? "" : "s"} late` : "";
          const title = `${periodLabel(e.period)} — ${s.label}${late} · paid ${usd(e.amount_paid)} of ${usd(e.rent_charged)}`;
          return (
            <span
              key={e.period}
              className={`pay-cell ${s.cls}`}
              title={title}
              aria-label={title}
            />
          );
        })}
      </div>

      <div className="pay-legend">
        {STATUS_ORDER.map((st) => {
          const s = STATUS_STYLE[st];
          return (
            <span key={st} className="pay-legend-item">
              <span className={`pay-cell ${s.cls}`} aria-hidden />
              {s.label}
            </span>
          );
        })}
      </div>

      {spark && (
        <div className="pay-spark" title="Rolling balance owed over the last 5 years">
          <div className="eyebrow" style={{ marginTop: 4 }}>Balance trend</div>
          <svg
            viewBox={`0 0 ${spark.w} ${spark.h}`}
            preserveAspectRatio="none"
            className="spark-svg"
            role="img"
            aria-label={`Balance ranged from ${usd(spark.min)} to ${usd(spark.max)} over the period`}
          >
            <polyline className="spark-line" points={spark.points} />
          </svg>
          <div className="pay-spark-scale">
            <span>{usd(spark.min)}</span>
            <span>{usd(spark.max)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

interface Spark {
  points: string;
  w: number;
  h: number;
  min: number;
  max: number;
}

/** Map balance_after onto a normalized polyline. Returns null if < 2 points. */
function buildSparkline(entries: LedgerEntry[]): Spark | null {
  if (entries.length < 2) return null;
  const w = 100;
  const h = 28;
  const vals = entries.map((e) => e.balance_after);
  const min = Math.min(...vals, 0);
  const max = Math.max(...vals, 0);
  const span = max - min || 1;
  const step = w / (vals.length - 1);
  const points = vals
    .map((v, i) => {
      const x = i * step;
      // invert Y so higher balance sits higher on the chart
      const y = h - ((v - min) / span) * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return { points, w, h, min, max };
}
