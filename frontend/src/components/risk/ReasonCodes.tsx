import { ArrowDown, ArrowUp } from "lucide-react";
import type { ReasonCode } from "../../types";

/**
 * Top drivers of the estimate, split into "Raises" and "Lowers" groups. Each
 * row carries a direction icon AND the word Raises/Lowers AND a signed value,
 * so meaning never depends on colour. Bar length = |contribution| normalised
 * to the strongest driver on the card.
 */
export function ReasonCodes({ codes }: { codes: ReasonCode[] }) {
  if (!codes || codes.length === 0) {
    return (
      <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
        No individual drivers stood out for this applicant.
      </p>
    );
  }

  const maxAbs = Math.max(...codes.map((c) => Math.abs(c.contribution)), 1e-6);
  const raises = codes.filter((c) => c.direction === "increases");
  const lowers = codes.filter((c) => c.direction === "decreases");

  return (
    <div style={{ display: "grid", gap: 12, marginTop: 12 }}>
      {raises.length > 0 && (
        <Group title="Raises risk" tone="bad" codes={raises} maxAbs={maxAbs} />
      )}
      {lowers.length > 0 && (
        <Group title="Lowers risk" tone="good" codes={lowers} maxAbs={maxAbs} />
      )}
    </div>
  );
}

function Group({
  title,
  tone,
  codes,
  maxAbs,
}: {
  title: string;
  tone: "good" | "bad";
  codes: ReasonCode[];
  maxAbs: number;
}) {
  const up = tone === "bad";
  return (
    <div>
      <div
        className="icon-line"
        style={{
          fontSize: 11,
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.06em",
          color: up ? "var(--bad-text)" : "var(--good-text)",
          marginBottom: 6,
        }}
      >
        {up ? <ArrowUp size={13} aria-hidden /> : <ArrowDown size={13} aria-hidden />}
        {title}
      </div>
      <div style={{ display: "grid", gap: 7 }}>
        {codes.map((c) => {
          const width = Math.round((Math.abs(c.contribution) / maxAbs) * 100);
          const sign = up ? "+" : "−";
          return (
            <div key={c.feature} className="fh-row" style={{ margin: 0 }}>
              <div className="fh-head" style={{ marginBottom: 3 }}>
                <span>
                  <span className="sr-only">{up ? "Raises: " : "Lowers: "}</span>
                  {c.label}
                </span>
                <span className="fh-val" style={{ fontVariantNumeric: "tabular-nums" }}>
                  {sign}
                  {Math.abs(c.contribution).toFixed(2)}
                </span>
              </div>
              <div className="meter">
                <i className={up ? "bad" : "good"} style={{ width: `${width}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
