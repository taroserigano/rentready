import {
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
} from "recharts";

export const LABELS: Record<string, string> = {
  affordability: "Afford",
  area: "Area",
  amenities: "Amenities",
  bedrooms: "Beds",
  budget_pref: "Budget",
  bathrooms: "Baths",
  transit: "Transit",
  square_feet: "Size",
  parking: "Parking",
  balcony: "Balcony",
  laundry: "Laundry",
};

export function SignalRadar({
  breakdown,
}: {
  breakdown: Record<string, number>;
}) {
  const data = Object.entries(breakdown).map(([k, v]) => ({
    signal: LABELS[k] ?? k,
    value: Math.round(v * 100),
  }));
  if (data.length < 3) return null;
  return (
    <ResponsiveContainer width="100%" height={200}>
      <RadarChart data={data} outerRadius="70%">
        <PolarGrid stroke="var(--chart-grid)" />
        <PolarAngleAxis
          dataKey="signal"
          tick={{ fill: "var(--chart-axis)", fontSize: 11 }}
        />
        <Radar
          dataKey="value"
          stroke="var(--chart-1)"
          fill="var(--chart-1)"
          fillOpacity={0.3}
          isAnimationActive={false}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
