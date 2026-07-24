import type { HorizonPoint } from "../../types";

/** Each model head is a CUMULATIVE "at least one late payment by month T"
 * probability — mathematically non-decreasing, so plotting it always
 * produces a rising line regardless of the real risk shape. These labels
 * instead describe the WINDOW each point's incremental (marginal) risk
 * covers, so the chart reads as "how risky is this specific stretch of
 * time," which can rise, fall, or plateau. */
const WINDOW_LABELS: Record<string, string> = {
  late_1m: "Month 1",
  late_3m: "Months 2–3",
  late_6m: "Months 4–6",
  late_12m: "Months 7–12",
};

/** A compact single-series trend: the MARGINAL late-payment risk added in
 * each future window (not the cumulative "by this point" figure, which only
 * ever rises) — makes the forward-looking prediction visible without a chat
 * question. Single series, so no legend (the title names it); all 4 points
 * direct-labeled since that's under the "≤4 labels" ceiling. */
export function HorizonForecastTrend({ points }: { points: HorizonPoint[] }) {
  if (!points.length) return null;

  const width = 100;
  const height = 46;
  const padTop = 8;
  const padBottom = 6;
  const plotH = height - padTop - padBottom;
  const values = points.map((p) => p.incremental_probability ?? 0);
  const maxV = Math.max(0.1, ...values); // a floor so a near-zero trend isn't all noise
  const x = (i: number) => ((i + 0.5) / points.length) * width;
  const y = (v: number) => padTop + plotH * (1 - v / maxV);

  const linePath = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(2)} ${y(p.incremental_probability ?? 0).toFixed(2)}`)
    .join(" ");
  const areaPath =
    `${linePath} L ${x(points.length - 1).toFixed(2)} ${(padTop + plotH).toFixed(2)} ` +
    `L ${x(0).toFixed(2)} ${(padTop + plotH).toFixed(2)} Z`;

  return (
    <div className="horizon-trend">
      <div className="horizon-trend-head">
        <span className="eyebrow">Late-payment forecast</span>
        <span className="muted" style={{ fontSize: 12 }}>
          added risk per period, this property (not cumulative)
        </span>
      </div>
      <div className="horizon-trend-plot">
        {/* SVG draws only the line/area — safe to stretch non-uniformly.
            Dots are a separate HTML overlay (fixed pixel size, % position)
            so preserveAspectRatio="none" never turns circles into ellipses. */}
        <svg
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
          className="horizon-trend-svg"
          aria-hidden="true"
        >
          <path d={areaPath} className="horizon-trend-area" />
          <path d={linePath} className="horizon-trend-line" />
        </svg>
        <div
          className="horizon-trend-dots"
          role="img"
          aria-label={`Late-payment risk added per period: ${points
            .map((p) => `${WINDOW_LABELS[p.horizon] ?? p.label} ${p.incremental_probability != null ? Math.round(p.incremental_probability * 100) : "unknown"}%`)
            .join(", ")}`}
        >
          {points.map((p, i) => (
            <span
              key={p.horizon}
              className="horizon-trend-dot"
              style={{
                left: `${(x(i) / width) * 100}%`,
                top: `${(y(p.incremental_probability ?? 0) / height) * 100}%`,
              }}
              title={`${WINDOW_LABELS[p.horizon] ?? p.label}: ${p.incremental_probability != null ? `${Math.round(p.incremental_probability * 100)}% added risk` : "no data"} (${p.avg_probability != null ? `${Math.round(p.avg_probability * 100)}% cumulative by end of this window` : "n/a"})`}
            />
          ))}
        </div>
      </div>
      <div className="horizon-trend-labels">
        {points.map((p) => (
          <div key={p.horizon} className="horizon-trend-label">
            <div className="horizon-trend-value">
              {p.incremental_probability != null ? `${Math.round(p.incremental_probability * 100)}%` : "—"}
            </div>
            <div className="horizon-trend-period">{WINDOW_LABELS[p.horizon] ?? p.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
