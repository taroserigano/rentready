import { useEffect, useState } from "react";
import { ArrowRight, Info } from "lucide-react";
import { getRisk } from "../../api";
import type { RiskResult } from "../../types";
import { RiskGauge } from "./RiskGauge";
import { ReasonCodes } from "./ReasonCodes";
import {
  BAND_LABEL,
  BAND_TONE,
  RISK_DISCLAIMER,
  formatProbRange,
} from "./riskTone";

/**
 * Inline decision-support disclaimer. Rendered on the Risk page banner
 * (variant="banner", an `.alert.warn`) and on every RiskCard (variant="inline",
 * a muted footnote). `onViewModelCard` wires the "View model card" affordance.
 */
export function RiskDisclaimer({
  variant = "inline",
  onViewModelCard,
  text = RISK_DISCLAIMER,
}: {
  variant?: "banner" | "inline";
  onViewModelCard?: () => void;
  text?: string;
}) {
  const link = onViewModelCard && (
    <button
      type="button"
      className="linklike"
      onClick={onViewModelCard}
      style={{ marginLeft: 4 }}
    >
      View model card
    </button>
  );

  if (variant === "banner") {
    return (
      <div className="alert warn" role="note">
        <Info size={16} aria-hidden style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          {text} {link}
        </div>
      </div>
    );
  }

  return (
    <p className="muted" style={{ fontSize: 11, marginTop: 12, lineHeight: 1.5 }}>
      {RISK_DISCLAIMER} {link}
    </p>
  );
}

type RiskCardProps = {
  sectionNumber?: string;
  /** "Open risk detail →" — deep-link to the full Risk page. */
  onOpenFull?: () => void;
  /** Wires "How this is scored →" / the disclaimer's model-card link. */
  onViewModelCard?: () => void;
} & (
  | { result: RiskResult; applicantId?: undefined }
  | { applicantId: string; result?: undefined }
);

/**
 * Reusable late-payment risk panel — just a `.card`, so it embeds anywhere.
 * Either takes a `result` directly or self-fetches from an `applicantId`.
 */
export function RiskCard(props: RiskCardProps) {
  const { sectionNumber, onOpenFull, onViewModelCard } = props;
  const [fetched, setFetched] = useState<RiskResult | null>(null);
  const [error, setError] = useState("");
  const [showHow, setShowHow] = useState(false);

  const selfFetch = props.result === undefined;

  useEffect(() => {
    if (!selfFetch) return;
    let alive = true;
    setFetched(null);
    setError("");
    getRisk(props.applicantId!)
      .then((r) => alive && setFetched(r))
      .catch((e) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, [selfFetch, props.applicantId]);

  const result = props.result ?? fetched;

  if (error) {
    return (
      <div className="card">
        <h2>{sectionNumber ? `${sectionNumber} ` : ""}Late-payment risk</h2>
        <div className="error">{error}</div>
      </div>
    );
  }
  if (!result) {
    if (!selfFetch) return null;
    return (
      <div className="card">
        <h2>{sectionNumber ? `${sectionNumber} ` : ""}Late-payment risk</h2>
        <p className="muted" style={{ margin: 0 }}>
          Scoring…
        </p>
      </div>
    );
  }

  const tone = BAND_TONE[result.band];

  return (
    <div className="card">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: 12,
        }}
      >
        <div>
          <h2 style={{ marginBottom: 4 }}>
            {sectionNumber ? `${sectionNumber} ` : ""}Late-payment risk
          </h2>
          <p className="muted" style={{ margin: 0, fontSize: 13 }}>
            Advisory estimate — <strong>routes Elevated cases to human review</strong>,
            never an auto-decision.
          </p>
        </div>
        <span className={`badge tone-${tone}`}>{BAND_LABEL[result.band]}</span>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(150px, 200px) 1fr",
          gap: 18,
          alignItems: "center",
          marginTop: 14,
        }}
      >
        <div>
          <RiskGauge probability={result.probability} band={result.band} />
          <div
            style={{
              textAlign: "center",
              fontSize: 12,
              color: "var(--text-2)",
              marginTop: 4,
            }}
          >
            {formatProbRange(result.probability, result.range)}
          </div>
          <div
            style={{
              display: "flex",
              gap: 6,
              justifyContent: "center",
              flexWrap: "wrap",
              marginTop: 8,
            }}
          >
            <span
              className={`badge tone-${result.confidence === "high" ? "good" : "warn"}`}
            >
              {result.confidence === "high" ? "High confidence" : "Low confidence"}
            </span>
            <span
              className="badge tone-info"
              title={`Scored by the ${result.model_type} ${result.source}`}
            >
              {result.source === "model" ? result.model_type : "heuristic"}
            </span>
          </div>
        </div>
        <div>
          <div className="eyebrow">Key factors</div>
          <ReasonCodes codes={result.reason_codes} />
        </div>
      </div>

      <div
        style={{
          display: "flex",
          gap: 14,
          flexWrap: "wrap",
          marginTop: 14,
          alignItems: "center",
        }}
      >
        <button
          type="button"
          className="linklike"
          onClick={() =>
            onViewModelCard ? onViewModelCard() : setShowHow((s) => !s)
          }
        >
          How this is scored →
        </button>
        {onOpenFull && (
          <button type="button" className="linklike" onClick={onOpenFull}>
            Open risk detail <ArrowRight size={12} aria-hidden />
          </button>
        )}
      </div>

      {showHow && !onViewModelCard && (
        <p className="muted" style={{ fontSize: 12, marginTop: 8, lineHeight: 1.55 }}>
          A gradient-boosted model estimates the calibrated probability of a late
          payment from legitimate financial and payment factors only. Bands:{" "}
          <strong>Low</strong> below 15%, <strong>Moderate</strong> 15–40%,{" "}
          <strong>Elevated</strong> at or above 40%. Missing data lowers
          confidence and never pushes risk up. Trained on synthetic data.
        </p>
      )}

      <RiskDisclaimer onViewModelCard={onViewModelCard} />
    </div>
  );
}
