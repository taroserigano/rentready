import type { ReactNode } from "react";
import { AlertTriangle, CalendarClock, DollarSign, Info } from "lucide-react";
import type {
  ResidentArrearsPrediction,
  ResidentChurnPrediction,
  ResidentClassPrediction,
  ResidentSeriousPrediction,
} from "../../types";
import { RiskGauge } from "../risk/RiskGauge";
import { ReasonCodes } from "../risk/ReasonCodes";
import { BAND_LABEL, BAND_TONE, churnLabel, churnTone, formatProbRange, pct, usd } from "./residentsTone";

/** Shared confidence + source chips row (mirrors RiskCard). */
function MetaChips({
  confidence,
  source,
  model_type,
}: {
  confidence: "high" | "low";
  source: string;
  model_type?: string;
}) {
  return (
    <div style={{ display: "flex", gap: 6, justifyContent: "center", flexWrap: "wrap", marginTop: 8 }}>
      <span className={`badge tone-${confidence === "high" ? "good" : "warn"}`}>
        {confidence === "high" ? "High confidence" : "Low confidence"}
      </span>
      <span
        className="badge tone-info"
        title={`Scored by the ${model_type ?? source} ${source}`}
      >
        {source === "model" ? model_type ?? "model" : "heuristic"}
      </span>
    </div>
  );
}

/** A calibrated-probability card (late OR serious) — gauge + reason codes. */
function ClassPredictionCard({
  title,
  subtitle,
  gaugeLabel,
  pred,
  headerBadge,
}: {
  title: string;
  subtitle: ReactNode;
  gaugeLabel: string;
  pred: ResidentClassPrediction;
  headerBadge?: ReactNode;
}) {
  return (
    <div className="card" style={{ marginTop: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div>
          <h2 style={{ marginBottom: 4 }}>{title}</h2>
          <p className="muted" style={{ margin: 0, fontSize: 13 }}>
            {subtitle}
          </p>
        </div>
        {headerBadge ?? (
          <span className={`badge tone-${BAND_TONE[pred.band]}`}>{BAND_LABEL[pred.band]}</span>
        )}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(140px, 190px) 1fr",
          gap: 18,
          alignItems: "center",
          marginTop: 14,
        }}
      >
        <div>
          <RiskGauge probability={pred.probability} band={pred.band} />
          <div style={{ textAlign: "center", fontSize: 12, color: "var(--text-2)", marginTop: 4 }}>
            {gaugeLabel}: {formatProbRange(pred.probability, pred.range)}
          </div>
          <MetaChips confidence={pred.confidence} source={pred.source} model_type={pred.model_type} />
        </div>
        <div>
          <div className="eyebrow">Key factors</div>
          <ReasonCodes codes={pred.reason_codes} />
        </div>
      </div>
    </div>
  );
}

/** 1 — Late payment next quarter. */
export function LatePredictionCard({ pred }: { pred: ResidentClassPrediction }) {
  return (
    <ClassPredictionCard
      title="Late payment — next quarter"
      subtitle="Estimated chance of any late/partial/missed payment in the next 3 months. Use it to prioritise proactive outreach — never an automated action."
      gaugeLabel="Late risk"
      pred={pred}
    />
  );
}

/** 2 — Serious delinquency; always routes to a human reviewer. */
export function SeriousPredictionCard({ pred }: { pred: ResidentSeriousPrediction }) {
  return (
    <ClassPredictionCard
      title="Serious delinquency — next quarter"
      subtitle="Chance of 30+ days late or a full month in arrears. Flagged cases are routed to a person to review, not acted on automatically."
      gaugeLabel="Serious risk"
      pred={pred}
      headerBadge={
        <span className="badge tone-bad" title="Routed to human review">
          <AlertTriangle size={12} aria-hidden /> Routes to review
        </span>
      }
    />
  );
}

/** 3 — Expected arrears (regression $) with a prediction interval. */
export function ArrearsPredictionCard({ pred }: { pred: ResidentArrearsPrediction }) {
  const [lo, hi] = pred.interval ?? [pred.expected_balance, pred.expected_balance];
  return (
    <div className="card" style={{ marginTop: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div>
          <h2 style={{ marginBottom: 4 }}>Expected arrears — end of next quarter</h2>
          <p className="muted" style={{ margin: 0, fontSize: 13 }}>
            Projected $ balance owed in 3 months. Supports outreach and payment-plan conversations.
          </p>
        </div>
        <span className="badge tone-info">
          <DollarSign size={12} aria-hidden /> Regression
        </span>
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
        <div style={{ textAlign: "center" }}>
          <div className="big-num" style={{ color: "var(--accent-text)" }}>
            {usd(pred.expected_balance)}
          </div>
          <div style={{ fontSize: 12, color: "var(--text-2)", marginTop: 4 }}>
            interval {usd(lo)} – {usd(hi)}
          </div>
          <div style={{ display: "flex", gap: 6, justifyContent: "center", flexWrap: "wrap", marginTop: 8 }}>
            <span className={`badge tone-${pred.confidence === "high" ? "good" : "warn"}`}>
              {pred.confidence === "high" ? "High confidence" : "Low confidence"}
            </span>
            <span className="badge tone-info" title={`Scored by ${pred.source}`}>
              {pred.source === "model" ? "model" : "heuristic"}
            </span>
          </div>
        </div>
        <div>
          <div className="eyebrow">Key factors</div>
          <ReasonCodes codes={pred.reason_codes} />
        </div>
      </div>
    </div>
  );
}

/** 4 — Churn (non-renewal). Renders "not applicable" when the lease isn't ending soon. */
export function ChurnPredictionCard({ pred }: { pred: ResidentChurnPrediction }) {
  const na = pred.band === "not_applicable" || pred.probability == null;

  return (
    <div className="card" style={{ marginTop: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div>
          <h2 style={{ marginBottom: 4 }}>Churn — non-renewal risk</h2>
          <p className="muted" style={{ margin: 0, fontSize: 13 }}>
            Chance the resident does not renew, scored only when the lease ends within 6 months.
          </p>
        </div>
        <span className={`badge tone-${churnTone(pred.band)}`}>{churnLabel(pred.band)}</span>
      </div>

      {na ? (
        <div className="alert" style={{ marginTop: 14, background: "var(--panel-2)", borderColor: "var(--line)", color: "var(--text-2)" }} role="note">
          <Info size={16} aria-hidden style={{ flexShrink: 0 }} />
          <div>
            Not applicable — lease is not ending within 6 months
            {Number.isFinite(pred.months_to_lease_end) && pred.months_to_lease_end > 0
              ? ` (about ${pred.months_to_lease_end} months to lease end).`
              : "."}{" "}
            Churn is only estimated inside the renewal window.
          </div>
        </div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(140px, 190px) 1fr",
            gap: 18,
            alignItems: "center",
            marginTop: 14,
          }}
        >
          <div>
            <RiskGauge probability={pred.probability ?? 0} band={pred.band === "not_applicable" ? "low" : pred.band} />
            <div style={{ textAlign: "center", fontSize: 12, color: "var(--text-2)", marginTop: 4 }}>
              Non-renewal: {pct(pred.probability)}
              <span className="icon-line" style={{ justifyContent: "center", marginTop: 2 }}>
                <CalendarClock size={12} aria-hidden /> {pred.months_to_lease_end} mo to lease end
              </span>
            </div>
            <div style={{ display: "flex", gap: 6, justifyContent: "center", flexWrap: "wrap", marginTop: 8 }}>
              <span className={`badge tone-${pred.confidence === "high" ? "good" : "warn"}`}>
                {pred.confidence === "high" ? "High confidence" : "Low confidence"}
              </span>
              <span className="badge tone-info" title={`Scored by ${pred.source}`}>
                {pred.source === "model" ? "model" : "heuristic"}
              </span>
            </div>
          </div>
          <div>
            <div className="eyebrow">Key factors</div>
            <ReasonCodes codes={pred.reason_codes} />
          </div>
        </div>
      )}
    </div>
  );
}
