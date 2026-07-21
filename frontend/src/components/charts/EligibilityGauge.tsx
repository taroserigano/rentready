/**
 * A simple horizontal gauge showing income-to-rent ratio against the 3x
 * policy threshold. Pure SVG (no chart lib needed).
 */
export function EligibilityGauge({ ratio }: { ratio: number | null }) {
  if (ratio == null) return null;
  const max = 5;
  const pct = Math.min(ratio / max, 1) * 100;
  const threshold = (3 / max) * 100;
  const color =
    ratio >= 3
      ? "var(--chart-2)"
      : ratio >= 2.5
        ? "var(--chart-3)"
        : "var(--chart-4)";
  return (
    <div style={{ marginTop: 12 }}>
      <div
        style={{
          position: "relative",
          height: 14,
          background: "var(--field)",
          borderRadius: 999,
          overflow: "hidden",
          border: "1px solid var(--line-strong)",
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: "100%",
            background: color,
            transition: "width 0.6s ease",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: `${threshold}%`,
            top: 0,
            bottom: 0,
            width: 2,
            background: "var(--text-2)",
          }}
          title="3x minimum"
        />
      </div>
      <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4 }}>
        income is <b style={{ color }}>{ratio.toFixed(1)}×</b> rent · 3× minimum (marker)
      </div>
    </div>
  );
}
