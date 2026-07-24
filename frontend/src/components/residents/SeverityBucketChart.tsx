import type { SeverityBucket } from "../../types";

const BUCKET_LABELS: Record<SeverityBucket["bucket"], string> = {
  "none": "On time",
  "1-29": "1–29 days",
  "30-59": "30–59 days",
  "60-89": "60–89 days",
  "90+": "90+ days",
};

/** Sequential, single-hue (light → dark) — this is ORDINAL severity data
 * (none < 1-29 < 30-59 < 60-89 < 90+), not independent categories, so one
 * hue ramping darker reads as "worse" without inventing new color meaning. */
const BUCKET_OPACITY: Record<SeverityBucket["bucket"], number> = {
  "none": 0.22,
  "1-29": 0.42,
  "30-59": 0.62,
  "60-89": 0.82,
  "90+": 1,
};

/** A single horizontal stacked bar: what share of residents fall in each
 * worst-delinquency bucket. A legend carries the labels (5 segments would
 * clutter direct labels on thin slices); a 2px surface gap separates
 * segments per the stacked-bar spec. */
export function SeverityBucketChart({ buckets }: { buckets: SeverityBucket[] }) {
  const total = buckets.reduce((sum, b) => sum + b.count, 0);
  if (total === 0) return null;

  return (
    <div className="severity-bar">
      <div className="severity-bar-head">
        <span className="eyebrow">Worst-delinquency forecast</span>
        <span className="muted" style={{ fontSize: 12 }}>
          share of residents by severity, next 12 months
        </span>
      </div>
      <div
        className="severity-bar-track"
        role="img"
        aria-label={`Worst delinquency distribution: ${buckets
          .map((b) => `${BUCKET_LABELS[b.bucket]} ${b.count} of ${total}`)
          .join(", ")}`}
      >
        {buckets.map((b) =>
          b.count > 0 ? (
            <div
              key={b.bucket}
              className="severity-bar-seg"
              style={{ flexGrow: b.count, opacity: BUCKET_OPACITY[b.bucket] }}
              title={`${BUCKET_LABELS[b.bucket]}: ${b.count} of ${total} residents`}
            />
          ) : null,
        )}
      </div>
      <div className="severity-bar-legend">
        {buckets.map((b) => (
          <div key={b.bucket} className="severity-bar-legend-item">
            <span className="severity-bar-swatch" style={{ opacity: BUCKET_OPACITY[b.bucket] }} />
            <span className="muted" style={{ fontSize: 12 }}>
              {BUCKET_LABELS[b.bucket]}
            </span>
            <span style={{ fontSize: 12, fontVariantNumeric: "tabular-nums" }}>{b.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
