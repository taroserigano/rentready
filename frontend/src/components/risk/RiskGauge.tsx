import { memo } from "react";
import {
  PolarAngleAxis,
  RadialBar,
  RadialBarChart,
  ResponsiveContainer,
} from "recharts";
import type { RiskBand } from "../../types";
import { BAND_COLOR, BAND_LABEL } from "./riskTone";

/**
 * Semicircular calibrated-probability gauge. The needle-free radial fill maps
 * 0–100% onto a 180° arc; the band label and percentage sit in the centre so
 * the reading never relies on colour alone. Static (isAnimationActive=false).
 * memo()'d: all props are primitives, so this only actually recharts-relayouts
 * when the probability/band/size themselves change, not on every parent render.
 */
export const RiskGauge = memo(function RiskGauge({
  probability,
  band,
  size = 168,
}: {
  probability: number;
  band: RiskBand;
  size?: number;
}) {
  const pct = Math.max(0, Math.min(100, Math.round(probability * 100)));
  const color = BAND_COLOR[band];
  const data = [{ name: "risk", value: pct }];

  return (
    <div
      role="img"
      aria-label={`Estimated late-payment probability ${pct} percent, ${BAND_LABEL[band]} band`}
      style={{ position: "relative", width: "100%", maxWidth: size, margin: "0 auto" }}
    >
      <ResponsiveContainer width="100%" height={size * 0.62}>
        <RadialBarChart
          data={data}
          startAngle={180}
          endAngle={0}
          innerRadius="72%"
          outerRadius="100%"
          barSize={14}
        >
          <PolarAngleAxis
            type="number"
            domain={[0, 100]}
            angleAxisId={0}
            tick={false}
          />
          <RadialBar
            dataKey="value"
            angleAxisId={0}
            cornerRadius={8}
            fill={color}
            background={{ fill: "var(--field)" }}
            isAnimationActive={false}
          />
        </RadialBarChart>
      </ResponsiveContainer>
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          bottom: 2,
          textAlign: "center",
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            fontSize: 30,
            fontWeight: 700,
            letterSpacing: "-0.02em",
            lineHeight: 1,
            fontVariantNumeric: "tabular-nums",
            color,
          }}
        >
          {pct}%
        </div>
        <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
          {BAND_LABEL[band]} risk
        </div>
      </div>
    </div>
  );
});
