import type { RiskModelCard } from "../../types";
import { BAND_LABEL } from "./riskTone";
import type { RiskBand } from "../../types";

/** Governance panel — intended use, feature policy, metrics, limitations. */
export function ModelCard({ card }: { card: RiskModelCard }) {
  return (
    <div className="card">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <h2 style={{ margin: 0 }}>Model card</h2>
        <span className="muted" style={{ fontSize: 12 }}>
          {card.name} · v{card.version}
          {card.trained_at
            ? ` · trained ${new Date(card.trained_at).toLocaleDateString()}`
            : ""}{" "}
          · {card.source}
        </span>
      </div>

      {card.description && (
        <p className="muted" style={{ marginTop: 8, fontSize: 13 }}>
          {card.description}
        </p>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: 12,
          marginTop: 12,
        }}
      >
        <div className="subpanel">
          <div className="subpanel-title">Intended use</div>
          <p style={{ fontSize: 13, margin: 0, lineHeight: 1.5 }}>
            {card.intended_use}
          </p>
          <p
            className="warn-text"
            style={{ fontSize: 12, marginTop: 8, marginBottom: 0 }}
          >
            Not for automated decisions. Estimates from synthetic data support a
            human reviewer — they do not approve, deny, price, or condition a lease.
          </p>
        </div>

        <div className="subpanel">
          <div className="subpanel-title">Test metrics</div>
          <Mini label="ROC-AUC" value={card.metrics.auc?.toFixed(3)} />
          <Mini label="PR-AUC" value={card.metrics.pr_auc?.toFixed(3)} />
          <Mini label="Brier (lower better)" value={card.metrics.brier?.toFixed(3)} />
          <Mini label="ECE (lower better)" value={card.metrics.ece?.toFixed(3)} />
          <Mini label="Test rows" value={card.metrics.n_test} />
        </div>

        <div className="subpanel">
          <div className="subpanel-title">Bands</div>
          {card.bands.map((b) => (
            <Mini
              key={b.band}
              label={BAND_LABEL[b.band as RiskBand] ?? b.band}
              value={`${Math.round(b.min * 100)}–${Math.round(b.max * 100)}%`}
            />
          ))}
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: 12,
          marginTop: 12,
        }}
      >
        <div className="subpanel">
          <div className="subpanel-title">Features used ({card.features.length})</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {card.features.map((f) => (
              <span key={f} className="badge tone-info">
                {f}
              </span>
            ))}
          </div>
        </div>

        <div className="subpanel">
          <div className="subpanel-title">
            Structurally excluded ({card.excluded.length})
          </div>
          <ul className="reasons" style={{ marginTop: 0 }}>
            {card.excluded.map((e) => (
              <li key={e.field} style={{ fontSize: 12 }}>
                <b>{e.field}</b> — {e.reason}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {card.limitations.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <h3 className="note-title">Limitations</h3>
          <ul className="reasons">
            {card.limitations.map((l, i) => (
              <li key={i} style={{ fontSize: 12 }}>
                {l}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function Mini({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="mini-row">
      <span className="k">{label}</span>
      <span className="v">{(value as string) ?? "—"}</span>
    </div>
  );
}
