import { useMemo, useState } from "react";
import { ArrowUpDown, Calculator, ExternalLink, TrendingDown, Wallet } from "lucide-react";
import type { PropertyRecommendation, RecommendResponse } from "../types";
import { LABELS, SignalRadar } from "./charts/SignalRadar";
import { Feedback } from "./Feedback";
import { isNum, money, shortDate } from "./PropertyDetail";
import { PropThumb } from "./PropPhoto";

/**
 * One compact line from the new listing fields, e.g.
 * "Built 2018 · Available Aug 1 · Deposit $1,450".
 * Renders nothing when the fields are missing (older cached rows).
 */
function RecFacts({ r }: { r: PropertyRecommendation }) {
  const avail = shortDate(r.availability_date);
  const parts = [
    isNum(r.year_built) ? `Built ${r.year_built}` : null,
    avail ? `Available ${avail}` : null,
    isNum(r.security_deposit) ? `Deposit ${money(r.security_deposit)}` : null,
  ].filter(Boolean);
  if (parts.length === 0) return null;
  return (
    <div className="meta" style={{ marginTop: 2 }}>
      {parts.join(" · ")}
    </div>
  );
}

type RecSort = "recommended" | "match" | "rent";

/**
 * "Why this rank?" — reconstructs the deterministic score from the same
 * weights the backend used: contribution_i = w_i·s_i / Σ(w present), which
 * sums to the shown score. Pure arithmetic; hidden if weights aren't present.
 */
function RecScoreMath({
  breakdown,
  weights,
}: {
  breakdown: Record<string, number>;
  weights?: Record<string, number>;
}) {
  const [open, setOpen] = useState(false);
  const rows = useMemo(() => {
    if (!weights) return [];
    const present = Object.keys(breakdown).filter((k) => weights[k] != null);
    const totalW = present.reduce((s, k) => s + weights[k], 0) || 1;
    return present
      .map((k) => ({
        key: k,
        label: LABELS[k] ?? k,
        sub: breakdown[k],
        contrib: (weights[k] * breakdown[k]) / totalW,
      }))
      .sort((a, b) => b.contrib - a.contrib);
  }, [breakdown, weights]);

  if (!weights || rows.length === 0) return null;
  return (
    <div style={{ marginTop: 8 }}>
      <button
        className="btn-small btn-ghost icon-line"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <Calculator size={12} /> {open ? "Hide" : "Why this rank?"}
      </button>
      {open && (
        <div className="score-math">
          {rows.map((r) => (
            <div className="score-row" key={r.key}>
              <span className="score-label">{r.label}</span>
              <span className="meter">
                <i
                  className={r.sub >= 0.7 ? "good" : r.sub >= 0.4 ? "warn" : "bad"}
                  style={{ width: `${Math.round(r.sub * 100)}%` }}
                />
              </span>
              <span className="score-contrib">
                +{Math.round(r.contrib * 100)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Rent as a share of income → {pct, tone}. Null when income is unknown. */
function rentBurden(rent: number, income?: number) {
  if (!income || income <= 0) return null;
  const pct = Math.round((rent / income) * 100);
  const tone = pct <= 30 ? "good" : pct <= 40 ? "warn" : "bad";
  return { pct, tone };
}

/** The single lowest sub-signal (< 0.5) — the honest trade-off to surface. */
function weakestSignal(breakdown: Record<string, number>) {
  const entries = Object.entries(breakdown ?? {});
  if (entries.length === 0) return null;
  const [key, val] = entries.reduce((a, b) => (b[1] < a[1] ? b : a));
  if (val >= 0.5) return null;
  return { label: LABELS[key] ?? key, pct: Math.round(val * 100) };
}

export function Recommendations({
  data,
  applicantId,
  monthlyIncome,
  onViewListing,
}: {
  data: RecommendResponse;
  applicantId?: string;
  /** Applicant income, for the per-rec rent-burden chip. */
  monthlyIncome?: number;
  /** Opens the dedicated full-page listing for a property id. */
  onViewListing?: (id: string) => void;
}) {
  const [sort, setSort] = useState<RecSort>("recommended");

  // Display-only reorder of a COPY. "recommended" keeps the deterministic
  // scorer order the backend returned — never mutate data.recommendations.
  const recs = useMemo(() => {
    const list = [...data.recommendations];
    if (sort === "match") list.sort((a, b) => b.score - a.score);
    else if (sort === "rent") list.sort((a, b) => a.monthly_rent - b.monthly_rent);
    return list;
  }, [data.recommendations, sort]);

  return (
    <div className="card">
      <div className="rec-head">
        <h2 style={{ margin: 0 }}>4. Recommended properties</h2>
        {data.recommendations.length > 1 && (
          <label className="rec-sort">
            <ArrowUpDown size={12} />
            <select value={sort} onChange={(e) => setSort(e.target.value as RecSort)}>
              <option value="recommended">Recommended</option>
              <option value="match">Best match</option>
              <option value="rent">Lowest rent</option>
            </select>
          </label>
        )}
      </div>
      <p className="muted">
        via GraphRAG · Graph DB: {data.graph_backend} · ranked by: {data.source}
        {data.relaxed && " · budget relaxed to show options"}
        {sort !== "recommended" && " · reordered for display; ranking unchanged"}
      </p>
      {data.recommendations.length === 0 && (
        <p className="muted">No properties matched the hard constraints.</p>
      )}
      {recs.map((r) => {
        const burden = rentBurden(r.monthly_rent, monthlyIncome);
        const weak = weakestSignal(r.signal_breakdown);
        return (
          <div className="rec" key={r.property_id}>
            <PropThumb
              src={r.photo_url || r.photo_urls?.[0]}
              alt={r.name}
              className="rec-thumb"
              phClassName="rec-thumb prop-img-ph"
            />
            <div className="top">
              <span className="name">
                {r.name}{" "}
                <span className="muted" style={{ fontSize: 12 }}>
                  {Math.round(r.score * 100)}% match
                </span>
              </span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
                {burden && (
                  <span
                    className={`badge tone-${burden.tone} icon-line`}
                    title={`Rent is ${burden.pct}% of monthly income`}
                  >
                    <Wallet size={12} />
                    {burden.pct}% of income
                  </span>
                )}
                <span className="rent">${r.monthly_rent.toLocaleString()}/mo</span>
                {onViewListing && (
                  <button
                    className="btn-small btn-ghost"
                    style={{ display: "inline-flex", alignItems: "center", gap: 5 }}
                    onClick={() => onViewListing(r.property_id)}
                  >
                    <ExternalLink size={12} /> View listing
                  </button>
                )}
              </span>
            </div>
            <div className="meta">
              {r.property_type} · {r.area} · {r.bedrooms} bd · {r.bathrooms} ba ·{" "}
              {r.square_feet} sqft
              {[
                r.has_balcony && "balcony",
                r.in_unit_laundry && "in-unit laundry",
                r.parking_type && `${r.parking_type} parking`,
                r.walk_score != null && `walk ${r.walk_score}`,
              ]
                .filter(Boolean)
                .map((f) => ` · ${f}`)}
            </div>
            <RecFacts r={r} />
            <div className="reason">{r.match_reason}</div>
            {weak && (
              <div className="tradeoff">
                <TrendingDown size={12} />
                Weakest fit: <span className="signal">{weak.label} ({weak.pct}%)</span>
              </div>
            )}
            <SignalRadar breakdown={r.signal_breakdown} />
            <RecScoreMath breakdown={r.signal_breakdown} weights={data.weights} />
            {r.fit_highlights.length > 0 && (
              <div className="badges" style={{ marginTop: 8 }}>
                {r.fit_highlights.map((h, i) => (
                  <span key={i} className="badge tone-info">
                    {h}
                  </span>
                ))}
              </div>
            )}
            <div style={{ marginTop: 8 }}>
              <Feedback
                applicantId={applicantId}
                target="recommendation"
                itemId={r.property_id}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
