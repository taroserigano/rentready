import type { ArrearsBreakdownPoint } from "../../types";
import { usd } from "./residentsTone";

/** A compact 4-bar chart: expected total (cumulative) arrears balance at
 * each of the next 4 quarterly checkpoints, this property — a genuine
 * timeline (each quarter is its own trained head, not a proration), so the
 * bars are expected to trend upward as balances accumulate across the year.
 * Single series (magnitude), single hue, no legend needed; all 4 bars
 * direct-labeled (under the "≤4 labels" ceiling). */
export function ArrearsBreakdownChart({ points }: { points: ArrearsBreakdownPoint[] }) {
  if (!points.length) return null;
  const max = Math.max(1, ...points.map((p) => p.expected));

  return (
    <div className="arrears-bars">
      <div className="arrears-bars-head">
        <span className="eyebrow">Expected arrears</span>
        <span className="muted" style={{ fontSize: 12 }}>
          next 4 quarters, this property
        </span>
      </div>
      <div className="arrears-bars-plot" role="img" aria-label={`Expected arrears: ${points.map((p) => `${p.label} ${usd(p.expected)}`).join(", ")}`}>
        {points.map((p) => (
          <div key={p.key} className="arrears-bar-col" title={`${p.label}: ${usd(p.expected)}`}>
            <div className="arrears-bar-value">{usd(p.expected)}</div>
            <div className="arrears-bar-track">
              <div
                className="arrears-bar-fill"
                style={{ height: `${Math.max(3, (p.expected / max) * 100)}%` }}
              />
            </div>
            <div className="arrears-bar-label">{p.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
