import { AlertTriangle } from "lucide-react";
import type {
  BinaryHead,
  CountHead,
  MulticlassHead,
  RegressionHead,
  ResidentHead,
  SurvivalHead,
} from "../../types";
import { RiskGauge } from "../risk/RiskGauge";
import { ReasonCodes } from "../risk/ReasonCodes";
import {
  BAND_LABEL,
  BAND_TONE,
  FAMILY_HINT,
  FAMILY_LABEL,
  formatProbRange,
  headLabel,
  pct,
  usd,
} from "./residentsTone";

/**
 * Renders the v2 prediction HEADS grouped by FAMILY. Each family is a labelled
 * section; heads render by kind through the shared risk primitives (RiskGauge
 * for binary, value+interval tiles for count/regression, class-probability bars
 * for multiclass, a months-to-event tile + survival sparkline for survival).
 * The most salient head of each family exposes its TreeSHAP reason codes in a
 * collapsible "Why" panel, so the section stays scannable without hiding the
 * grounding. Pure + token-driven — no hardcoded colours, no fetching.
 *
 * Also used (compact) by the chat artifact to paint a resident's predictions.
 */

/** Family → the head whose reason codes best explain the family. */
const PRIMARY_HEAD: Record<string, string> = {
  late: "late_12m",
  frequency: "late_count_12m",
  severity: "delinquency_bucket_12m",
  arrears: "arrears_3m",
  cure: "p_cure_6m",
  retention: "churn",
};

/** Order families read in; anything unknown is appended after these. */
const FAMILY_ORDER = [
  "late",
  "frequency",
  "severity",
  "arrears",
  "cure",
  "retention",
];

/** True when a regression/count head is a dollar amount (not a day/month count). */
function isMoneyHead(name: string): boolean {
  return /arrears|balance/.test(name);
}

/** Headline value for a count/regression head (money, days, or a plain count). */
function scalarText(name: string, value: number): string {
  if (isMoneyHead(name)) return usd(value);
  if (name === "max_days_late_12m") {
    return `${Math.round(value)} day${Math.round(value) === 1 ? "" : "s"}`;
  }
  const rounded = Math.round(value * 10) / 10;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}

// --- per-kind tiles ---------------------------------------------------------

/** binary → a compact gauge with label, band and calibration range. */
function BinaryTile({ name, head }: { name: string; head: BinaryHead }) {
  const serious = name === "serious";
  return (
    <div className="head-tile">
      <div className="head-tile-title">
        {headLabel(name)}
        {serious && (
          <span className="badge tone-bad icon-line" title="Routed to human review">
            <AlertTriangle size={11} aria-hidden /> Review
          </span>
        )}
      </div>
      <RiskGauge probability={head.probability} band={head.band} size={132} />
      <div className="head-tile-sub">{formatProbRange(head.probability, head.range)}</div>
      <span className={`badge tone-${BAND_TONE[head.band]}`}>{BAND_LABEL[head.band]}</span>
    </div>
  );
}

/** count / regression → a big number with its prediction interval. */
function ScalarTile({
  name,
  head,
}: {
  name: string;
  head: CountHead | RegressionHead;
}) {
  const [lo, hi] = head.interval ?? [head.expected, head.expected];
  return (
    <div className="head-tile">
      <div className="head-tile-title">{headLabel(name)}</div>
      <div className="big-num" style={{ color: "var(--accent-text)" }}>
        {scalarText(name, head.expected)}
      </div>
      <div className="head-tile-sub">
        interval {scalarText(name, lo)} – {scalarText(name, hi)}
      </div>
    </div>
  );
}

/** multiclass → labelled probability bars, predicted bucket highlighted. */
function MulticlassTile({ name, head }: { name: string; head: MulticlassHead }) {
  const entries = Object.entries(head.class_probs ?? {});
  const max = Math.max(...entries.map(([, v]) => v), 1e-6);
  return (
    <div className="head-tile head-tile--wide">
      <div className="head-tile-title">
        {headLabel(name)}
        {head.predicted_bucket && (
          <span className="badge tone-info">{head.predicted_bucket}</span>
        )}
      </div>
      <div style={{ display: "grid", gap: 7, width: "100%", marginTop: 4 }}>
        {entries.map(([label, p]) => {
          const on = label === head.predicted_bucket;
          return (
            <div key={label} className="fh-row" style={{ margin: 0 }}>
              <div className="fh-head" style={{ marginBottom: 3 }}>
                <span style={on ? { fontWeight: 600, color: "var(--text)" } : undefined}>
                  {label}
                </span>
                <span className="fh-val" style={{ fontVariantNumeric: "tabular-nums" }}>
                  {pct(p)}
                </span>
              </div>
              <div className="meter">
                <i
                  className={on ? "warn" : "info"}
                  style={{ width: `${Math.round((p / max) * 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** survival → median/expected time-to-event + a small survival-curve line. */
function SurvivalTile({ name, head }: { name: string; head: SurvivalHead }) {
  const months = head.median_months ?? head.expected_months ?? null;
  const curve = head.survival_curve ?? [];
  const spark = buildCurve(curve);
  return (
    <div className="head-tile head-tile--wide">
      <div className="head-tile-title">{headLabel(name)}</div>
      <div className="big-num" style={{ color: "var(--accent-text)" }}>
        {months == null ? "—" : `${Math.round(months)} mo`}
      </div>
      <div className="head-tile-sub">
        {head.median_months == null
          ? "unlikely to clear within the horizon"
          : "median months to clear the balance"}
      </div>
      {spark && (
        <svg
          viewBox={`0 0 ${spark.w} ${spark.h}`}
          preserveAspectRatio="none"
          className="spark-svg"
          role="img"
          aria-label="Probability the balance is still outstanding, month by month"
          style={{ marginTop: 6 }}
        >
          <polyline className="spark-line" points={spark.points} />
        </svg>
      )}
    </div>
  );
}

interface Curve {
  points: string;
  w: number;
  h: number;
}

/** Map a 0..1 survival curve onto a normalized polyline (null if < 2 points). */
function buildCurve(vals: number[]): Curve | null {
  if (!vals || vals.length < 2) return null;
  const w = 100;
  const h = 24;
  const step = w / (vals.length - 1);
  const points = vals
    .map((v, i) => {
      const clamped = Math.max(0, Math.min(1, v));
      return `${(i * step).toFixed(1)},${(h - clamped * h).toFixed(1)}`;
    })
    .join(" ");
  return { points, w, h };
}

/** Dispatch one head to its kind-specific tile. */
function HeadTile({ name, head }: { name: string; head: ResidentHead }) {
  switch (head.kind) {
    case "binary":
      return <BinaryTile name={name} head={head} />;
    case "count":
    case "regression":
      return <ScalarTile name={name} head={head} />;
    case "multiclass":
      return <MulticlassTile name={name} head={head} />;
    case "survival":
      return <SurvivalTile name={name} head={head} />;
    default:
      return null;
  }
}

// --- family section ---------------------------------------------------------

function FamilySection({
  family,
  names,
  heads,
  compact,
}: {
  family: string;
  names: string[];
  heads: Record<string, ResidentHead>;
  compact?: boolean;
}) {
  const present = names.filter((n) => heads[n]);
  if (present.length === 0) return null;

  const primaryName = PRIMARY_HEAD[family];
  const primary =
    (primaryName && heads[primaryName]) || heads[present[0]] || null;
  const reasons = primary?.reason_codes ?? [];

  const header = (
    <div className="fam-head">
      <h3 style={{ margin: 0 }}>{FAMILY_LABEL[family] ?? family}</h3>
      {!compact && FAMILY_HINT[family] && (
        <p className="muted" style={{ margin: "2px 0 0", fontSize: 13 }}>
          {FAMILY_HINT[family]}
        </p>
      )}
    </div>
  );

  const grid = (
    <div className="head-grid">
      {present.map((n) => (
        <HeadTile key={n} name={n} head={heads[n]} />
      ))}
    </div>
  );

  const why = reasons.length > 0 && (
    <details className="fam-why">
      <summary>Why — key factors</summary>
      <ReasonCodes codes={reasons} />
    </details>
  );

  if (compact) {
    return (
      <div className="subpanel fam-section" style={{ marginTop: 0 }}>
        {header}
        {grid}
        {why}
      </div>
    );
  }
  return (
    <section className="card fam-section" style={{ marginTop: 0 }}>
      {header}
      {grid}
      {why}
    </section>
  );
}

export function ResidentFamilies({
  heads,
  families,
  compact,
}: {
  heads: Record<string, ResidentHead>;
  families: Record<string, string[]>;
  compact?: boolean;
}) {
  // Order the known families first, then any extras the backend may add.
  const keys = [
    ...FAMILY_ORDER.filter((f) => families[f]?.length),
    ...Object.keys(families).filter(
      (f) => !FAMILY_ORDER.includes(f) && families[f]?.length,
    ),
  ];
  if (keys.length === 0) return null;

  return (
    <div style={{ display: "grid", gap: compact ? 12 : 18 }}>
      {keys.map((f) => (
        <FamilySection
          key={f}
          family={f}
          names={families[f]}
          heads={heads}
          compact={compact}
        />
      ))}
    </div>
  );
}
