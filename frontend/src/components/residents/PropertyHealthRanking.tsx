import { useEffect, useState } from "react";
import { ArrowRight, Award } from "lucide-react";
import { getResidentsHealth } from "../../api";
import type { PropertyHealth } from "../../types";
import { gradeTone } from "./residentsTone";

function errText(e: unknown): string {
  if (e instanceof TypeError) return "Could not reach the server. Is the backend running?";
  if (e instanceof Error && e.message) return e.message;
  return "Could not load the property-health ranking.";
}

/** Clamp a 0–100 score to a meter width. */
function scorePct(score: number): number {
  return Math.max(0, Math.min(100, Math.round(score)));
}

/**
 * One row of the ranking: grade chip + property name + health meter + driver.
 * Meaning never rests on colour — the grade letter and the score always show,
 * and the meter tone is paired with them. Optionally clickable to drill in.
 */
export function HealthRow({
  item,
  rank,
  onSelect,
}: {
  item: PropertyHealth;
  rank?: number;
  onSelect?: (propertyId: string) => void;
}) {
  const tone = gradeTone(item.grade);
  const pctW = scorePct(item.score);
  const clickable = !!onSelect;
  const Cell = clickable ? "button" : "div";
  return (
    <Cell
      type={clickable ? "button" : undefined}
      className={`health-row${clickable ? " is-clickable" : ""}`}
      onClick={clickable ? () => onSelect!(item.property_id) : undefined}
    >
      <span className={`grade-chip tone-${tone}`} aria-label={`Grade ${item.grade}`}>
        {item.grade}
      </span>
      <div className="health-row-body">
        <div className="fh-head" style={{ marginBottom: 3 }}>
          <span className="health-row-name">
            {rank != null && <span className="muted">{rank}. </span>}
            {item.name || item.property_id}
          </span>
          <span className="fh-val" style={{ fontVariantNumeric: "tabular-nums" }}>
            {pctW}
            <span className="muted" style={{ fontSize: 12 }}> / 100</span>
          </span>
        </div>
        <div className="meter">
          <i className={tone === "info" ? undefined : tone} style={{ width: `${pctW}%` }} />
        </div>
        <div className="health-row-meta">
          <span>
            {item.resident_count} resident{item.resident_count === 1 ? "" : "s"}
          </span>
          {item.top_driver && (
            <span className="secondary" title="Strongest driver of this score">
              · {item.top_driver}
            </span>
          )}
        </div>
      </div>
      {clickable && <ArrowRight size={16} className="health-row-go" aria-hidden />}
    </Cell>
  );
}

/** Compact, self-contained health list — reused by the chat artifact. */
export function HealthList({
  items,
  onSelect,
  limit,
}: {
  items: PropertyHealth[];
  onSelect?: (propertyId: string) => void;
  limit?: number;
}) {
  const rows = limit ? items.slice(0, limit) : items;
  return (
    <div className="health-list">
      {rows.map((item, i) => (
        <HealthRow key={item.property_id} item={item} rank={i + 1} onSelect={onSelect} />
      ))}
    </div>
  );
}

/**
 * The regional-director view: the portfolio's properties ranked best→worst by a
 * composite health score (0–100, letter grade A–F). Shown on the Residents page
 * when no property is selected, in place of the bare "pick a property" prompt.
 * Selecting a row drills into that property's residents.
 */
export function PropertyHealthRanking({
  onSelect,
}: {
  onSelect?: (propertyId: string) => void;
}) {
  const [items, setItems] = useState<PropertyHealth[] | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError("");
    getResidentsHealth()
      .then((r) => alive && setItems(r))
      .catch((e) => alive && setError(errText(e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="card">
        <p className="muted" style={{ margin: 0 }}>Ranking properties by health…</p>
      </div>
    );
  }

  // Graceful fallback: if health isn't available (older backend), keep the
  // original prompt so the page still guides the user to pick a property.
  if (error || !items || items.length === 0) {
    return (
      <div className="card">
        <p className="muted" style={{ margin: 0 }}>
          Select a property above to view its residents and their forward-looking risk.
        </p>
      </div>
    );
  }

  const best = items[0];
  const worst = items[items.length - 1];

  return (
    <div className="card">
      <div className="rec-head">
        <h2 style={{ margin: 0 }}>Portfolio health</h2>
      </div>
      <p className="muted" style={{ margin: "4px 0 0", fontSize: 13 }}>
        Properties ranked healthiest → most at-risk, from a composite of predicted
        on-time payment, serious-delinquency, churn, and collection signals. Select
        one to review its residents.
      </p>

      {items.length > 1 && (
        <div className="health-callouts">
          <span className="badge tone-good icon-line" title="Healthiest property">
            <Award size={12} aria-hidden /> Best: {best.name || best.property_id} ({best.grade})
          </span>
          <span className="badge tone-bad" title="Most at-risk property">
            Needs attention: {worst.name || worst.property_id} ({worst.grade})
          </span>
        </div>
      )}

      <div style={{ marginTop: 14 }}>
        <HealthList items={items} onSelect={onSelect} />
      </div>
    </div>
  );
}
