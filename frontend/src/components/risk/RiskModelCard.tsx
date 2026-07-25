import { Cpu } from "lucide-react";
import type { RiskModelCard as RiskModelCardT } from "../../types";

/** Number → short string; leaves non-finite as an em dash. */
function num(v: unknown): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return Math.abs(v) < 1 ? v.toFixed(3) : v.toLocaleString("en-US");
}

/** "xgboost" -> "XGBoost", "heuristic" -> "Heuristic". */
function techLabel(modelType?: string): string {
  if (!modelType) return "Heuristic";
  if (modelType.toLowerCase() === "xgboost") return "XGBoost";
  if (modelType.toLowerCase() === "histgb") return "HistGradientBoosting";
  return modelType.charAt(0).toUpperCase() + modelType.slice(1);
}

/**
 * Governance panel for the applicant late-payment risk model. Rendered
 * defensively — the model-card payload is best-effort, so every section
 * guards its fields. Mirrors ResidentModelCard's layout/tone.
 */
export function RiskModelCard({ card }: { card: RiskModelCardT }) {
  const metrics = card.metrics ?? {};
  const bands = card.bands ?? [];
  const excluded = card.excluded ?? [];
  const features = card.features ?? [];
  const limitations = card.limitations ?? [];
  const isModel = card.source === "model";

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
        <h2 style={{ margin: 0 }}>Model Card — Late-Payment Risk</h2>
        <span className="muted" style={{ fontSize: 12 }}>
          {[card.name, card.version && `v${card.version}`, card.source].filter(Boolean).join(" · ")}
        </span>
      </div>

      <span
        className={`badge icon-line tone-${isModel ? "info" : "warn"}`}
        style={{ marginTop: 10 }}
        title={isModel ? "A trained model is scoring this — not a placeholder." : "Falling back to the transparent heuristic (no trained model loaded)."}
      >
        <Cpu size={12} aria-hidden />
        {techLabel(card.model_type)}
      </span>

      {card.description && (
        <p className="muted" style={{ marginTop: 8, fontSize: 13 }}>{card.description}</p>
      )}

      <div className="subpanel" style={{ marginTop: 12 }}>
        <div className="subpanel-title">Intended Use</div>
        <p style={{ fontSize: 13, margin: 0, lineHeight: 1.5 }}>
          {card.intended_use ??
            "Decision-support for a human reviewer. Estimates from synthetic data help prioritize attention."}
        </p>
        <p className="warn-text" style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }}>
          Not for automated approval, denial, pricing, or lease conditioning. Elevated scores route
          to a person for review.
        </p>
      </div>

      {Object.keys(metrics).length > 0 && (
        <div className="subpanel" style={{ marginTop: 12 }}>
          <div className="subpanel-title">Metrics</div>
          {Object.entries(metrics).map(([k, v]) => (
            <div className="mini-row" key={k}>
              <span className="k">{k}</span>
              <span className="v">{num(v)}</span>
            </div>
          ))}
        </div>
      )}

      {bands.length > 0 && (
        <div className="subpanel" style={{ marginTop: 12 }}>
          <div className="subpanel-title">Bands</div>
          {bands.map((b) => (
            <div className="mini-row" key={b.band}>
              <span className="k">{b.band}</span>
              <span className="v">
                {Math.round(b.min * 100)}%–{Math.round(b.max * 100)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {features.length > 0 && (
        <div className="subpanel" style={{ marginTop: 12 }}>
          <div className="mini-row">
            <span className="k">features used</span>
            <span className="v">{features.length}</span>
          </div>
        </div>
      )}

      {excluded.length > 0 && (
        <div className="subpanel" style={{ marginTop: 12 }}>
          <div className="subpanel-title">Structurally Excluded ({excluded.length})</div>
          <ul className="reasons" style={{ marginTop: 0 }}>
            {excluded.map((e) => (
              <li key={e.field} style={{ fontSize: 12 }}>
                <b>{e.field}</b> — {e.reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      {limitations.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <h3 className="note-title">Limitations</h3>
          <ul className="reasons">
            {limitations.map((l, i) => (
              <li key={i} style={{ fontSize: 12 }}>{l}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
