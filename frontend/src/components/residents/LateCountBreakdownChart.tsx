import type { LateCountBreakdownPoint } from "../../types";

/** 1 decimal for a fractional expected count; whole numbers stay bare. */
function count(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

/** A compact 4-bar chart: expected total late-payment COUNT at each of the
 * next 4 quarterly checkpoints, this property — a genuine cumulative-count
 * timeline (each quarter is its own trained head, not a proration), so the
 * bars are expected to trend upward across the year. Single series
 * (magnitude), single hue, no legend needed; all 4 bars direct-labeled
 * (under the "≤4 labels" ceiling). Mirrors ArrearsBreakdownChart's layout. */
export function LateCountBreakdownChart({ points }: { points: LateCountBreakdownPoint[] }) {
  if (!points.length) return null;
  const max = Math.max(1, ...points.map((p) => p.expected));

  return (
    <div className="late-count-bars">
      <div className="late-count-bars-head">
        <span className="eyebrow">Predicted late-payment count</span>
        <span className="muted" style={{ fontSize: 12 }}>
          next 4 quarters, this property
        </span>
      </div>
      <div
        className="late-count-bars-plot"
        role="img"
        aria-label={`Expected late-payment count: ${points.map((p) => `${p.label} ${count(p.expected)}`).join(", ")}`}
      >
        {points.map((p) => (
          <div key={p.key} className="late-count-bar-col" title={`${p.label}: ${count(p.expected)} expected late payments`}>
            <div className="late-count-bar-value">{count(p.expected)}</div>
            <div className="late-count-bar-track">
              <div
                className="late-count-bar-fill"
                style={{ height: `${Math.max(3, (p.expected / max) * 100)}%` }}
              />
            </div>
            <div className="late-count-bar-label">{p.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
