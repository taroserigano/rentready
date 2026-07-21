import type { ResidentModelCard as ResidentModelCardT, ResidentTargetCard } from "../../types";

/** Number → short string; leaves non-finite as an em dash. */
function num(v: unknown): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return Math.abs(v) < 1 ? v.toFixed(3) : v.toLocaleString("en-US");
}

const TARGET_LABEL: Record<string, string> = {
  late: "Late payment",
  arrears: "Expected arrears",
  churn: "Churn (non-renewal)",
  serious: "Serious delinquency",
};

function TargetPanel({ name, card }: { name: string; card: ResidentTargetCard }) {
  const metrics = card.metrics ?? {};
  const feats = card.feature_order ?? card.features ?? [];
  return (
    <div className="subpanel">
      <div className="subpanel-title">{TARGET_LABEL[name] ?? name}</div>
      {Object.keys(metrics).length > 0 ? (
        Object.entries(metrics).map(([k, v]) => (
          <div className="mini-row" key={k}>
            <span className="k">{k}</span>
            <span className="v">{num(v)}</span>
          </div>
        ))
      ) : (
        <p className="muted" style={{ fontSize: 12, margin: 0 }}>No metrics reported.</p>
      )}
      {feats.length > 0 && (
        <div className="mini-row">
          <span className="k">features</span>
          <span className="v">{feats.length}</span>
        </div>
      )}
      {(card.model_type || card.source) && (
        <div className="mini-row">
          <span className="k">model</span>
          <span className="v">{card.model_type ?? card.source}</span>
        </div>
      )}
    </div>
  );
}

/**
 * Governance panel for the four resident-risk models. Rendered defensively —
 * the model-card payload is best-effort, so every section guards its fields.
 */
export function ResidentModelCard({ card }: { card: ResidentModelCardT }) {
  const targets = card.targets ?? {};
  const excluded = card.excluded ?? [];
  const limitations = card.limitations ?? [];

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
        <h2 style={{ margin: 0 }}>Model card — resident risk</h2>
        <span className="muted" style={{ fontSize: 12 }}>
          {[card.name, card.version && `v${card.version}`, card.dgp_version, card.source]
            .filter(Boolean)
            .join(" · ")}
        </span>
      </div>

      {card.description && (
        <p className="muted" style={{ marginTop: 8, fontSize: 13 }}>{card.description}</p>
      )}

      <div className="subpanel" style={{ marginTop: 12 }}>
        <div className="subpanel-title">Intended use</div>
        <p style={{ fontSize: 13, margin: 0, lineHeight: 1.5 }}>
          {card.intended_use ??
            "Decision-support for proactive outreach and retention. Estimates from synthetic data help a person prioritise attention."}
        </p>
        <p className="warn-text" style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }}>
          Not for automated decisions. Never used for eviction, denial, pricing, or lease conditioning.
          Serious-delinquency flags route to a human reviewer.
        </p>
      </div>

      {Object.keys(targets).length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12, marginTop: 12 }}>
          {Object.entries(targets).map(([name, t]) => (
            <TargetPanel key={name} name={name} card={t} />
          ))}
        </div>
      )}

      {excluded.length > 0 && (
        <div className="subpanel" style={{ marginTop: 12 }}>
          <div className="subpanel-title">Structurally excluded ({excluded.length})</div>
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
