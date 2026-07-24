import { memo, useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { RiskBand, RiskRow } from "../../types";
import { BAND_COLOR, BAND_LABEL } from "./riskTone";

// Local copies of the shared Aurora chart tokens (mirrors Dashboard / Evaluations).
const AXIS_TICK = { fill: "var(--chart-axis)", fontSize: 11 };
const TOOLTIP_STYLE = {
  background: "var(--panel-2)",
  border: "1px solid var(--line-strong)",
  borderRadius: 8,
  fontSize: 12,
  color: "var(--text)",
} as const;
const TOOLTIP_CURSOR = { fill: "rgba(255,255,255,0.04)" };

const ORDER: RiskBand[] = ["low", "medium", "high"];

/* memo()'d + memoized `data`: Risk.tsx re-renders this on every search
 * keystroke/band-filter click even though `rows` itself hasn't changed, so
 * without this recharts was redoing its layout pass on every keystroke. */
export const RiskDistribution = memo(function RiskDistribution({ rows }: { rows: RiskRow[] }) {
  const data = useMemo(() => {
    const counts: Record<RiskBand, number> = { low: 0, medium: 0, high: 0 };
    for (const r of rows) counts[r.band] = (counts[r.band] ?? 0) + 1;
    return ORDER.map((b) => ({
      band: b,
      name: BAND_LABEL[b],
      count: counts[b],
      color: BAND_COLOR[b],
    }));
  }, [rows]);

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: -8 }}>
        <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
        <XAxis
          dataKey="name"
          tick={AXIS_TICK}
          axisLine={false}
          tickLine={false}
          interval={0}
        />
        <YAxis
          allowDecimals={false}
          tick={AXIS_TICK}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip cursor={TOOLTIP_CURSOR} contentStyle={TOOLTIP_STYLE} />
        <Bar
          dataKey="count"
          name="Applicants"
          barSize={40}
          radius={[4, 4, 0, 0]}
          isAnimationActive={false}
        >
          {data.map((d) => (
            <Cell key={d.band} fill={d.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
});
