import { useEffect, useState } from "react";
import { Lightbulb } from "lucide-react";
import { getStrength } from "../api";
import type { StrengthResult } from "../types";

const BAND_TONE: Record<string, string> = {
  strong: "good",
  solid: "warn",
  thin: "bad",
};

/** Advisory 0-100 tenant-quality score (F4). Separate from the pass/fail verdict. */
export function StrengthCard({
  applicantId,
  sectionNumber,
}: {
  applicantId: string;
  sectionNumber?: string;
}) {
  const [s, setS] = useState<StrengthResult | null>(null);
  useEffect(() => {
    let alive = true;
    getStrength(applicantId)
      .then((r) => alive && setS(r))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [applicantId]);

  if (!s) return null;
  const tone = BAND_TONE[s.band] ?? "info";

  return (
    <div className="card">
      <h2>{sectionNumber ? `${sectionNumber} ` : ""}Application strength</h2>
      <p className="muted">
        Advisory tenant-quality score — <strong>does not affect the eligibility verdict</strong>.
      </p>

      <div style={{ display: "flex", alignItems: "baseline", gap: 12, margin: "8px 0 12px" }}>
        <span style={{ fontSize: 34, fontWeight: 700, letterSpacing: "-0.02em" }}>
          {s.score}
          <span className="muted" style={{ fontSize: 16, fontWeight: 500 }}>/100</span>
        </span>
        <span className={`badge tone-${tone} strength-band`}>{s.band}</span>
      </div>

      <div style={{ display: "grid", gap: 6 }}>
        {s.factors.map((f) => (
          <div key={f.label} className="fh-row" style={{ margin: 0 }}>
            <div className="fh-head" style={{ marginBottom: 3 }}>
              <span>{f.label}</span>
              <span className="fh-val">
                {f.points}/{f.max}
              </span>
            </div>
            <div className="meter">
              <i
                className={f.points >= f.max * 0.75 ? "good" : f.points >= f.max * 0.4 ? "warn" : "bad"}
                style={{ width: `${(f.points / f.max) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {s.suggestions.length > 0 && (
        <div style={{ marginTop: 12, display: "grid", gap: 4 }}>
          {s.suggestions.map((t, i) => (
            <div key={i} className="icon-line muted" style={{ fontSize: 12 }}>
              <Lightbulb size={12} /> {t}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
