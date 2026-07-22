import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp, Info, Search } from "lucide-react";
import { getResidentModelCard, getResidentProperties, listPropertyResidents } from "../../api";
import type {
  PropertyResidentRollup,
  ResidentModelCard as ResidentModelCardT,
  ResidentPropertyOption,
  ResidentRow,
  RiskBand,
} from "../../types";
import { RiskDisclaimer } from "../risk/RiskCard";
import { ResidentDetail } from "./ResidentDetail";
import { ResidentModelCard } from "./ResidentModelCard";
import { BAND_LABEL, BAND_TONE, RESIDENT_DISCLAIMER, churnTone, pct, usd } from "./residentsTone";

const ROW_CAP = 25;

type SortKey = "unit" | "tenure" | "late" | "arrears" | "churn" | "serious" | "balance";
type BandFilter = "all" | RiskBand;

const BAND_FILTERS: { key: BandFilter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "low", label: BAND_LABEL.low },
  { key: "medium", label: BAND_LABEL.medium },
  { key: "high", label: BAND_LABEL.high },
];

function errText(e: unknown): string {
  if (e instanceof TypeError) return "Could not reach the server. Is the backend running?";
  if (e instanceof Error && e.message) return e.message;
  return "Could not reach the server. Is the backend running?";
}

/**
 * Residents page — a portfolio-wide, decision-support view of current-resident
 * risk across the 10 properties. A property selector, four portfolio KPI tiles,
 * a searchable/sortable/filterable resident table, and a full detail region.
 */
export function Residents({
  initialPropertyId,
  initialResidentId,
}: {
  initialPropertyId?: string;
  initialResidentId?: string;
}) {
  const [propOptions, setPropOptions] = useState<ResidentPropertyOption[] | null>(null);
  const [residents, setResidents] = useState<ResidentRow[] | null>(null);
  const [rollup, setRollup] = useState<PropertyResidentRollup | null>(null);
  const [card, setCard] = useState<ResidentModelCardT | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [listLoading, setListLoading] = useState(false);

  const [selectedProperty, setSelectedProperty] = useState<string>(initialPropertyId ?? "");
  const [showInfo, setShowInfo] = useState(false);
  const [selectedResident, setSelectedResident] = useState<string | null>(initialResidentId ?? null);

  const [query, setQuery] = useState("");
  const [bandFilter, setBandFilter] = useState<BandFilter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("late");
  const [sortAsc, setSortAsc] = useState(false);
  const [showAll, setShowAll] = useState(false);

  const modelCardRef = useRef<HTMLDivElement>(null);
  const scrollToModelCard = () =>
    modelCardRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });

  // Initial load is CHEAP: just the property picker (no scoring) + the model
  // card. Residents are fetched per-property, on selection (below).
  const load = useCallback(() => {
    setLoading(true);
    setError("");
    getResidentProperties()
      .then((r) => setPropOptions(r.properties))
      .catch((e) => setError(errText(e)))
      .finally(() => setLoading(false));
    getResidentModelCard().then(setCard).catch(() => {});
  }, []);

  useEffect(load, [load]);

  useEffect(() => {
    if (initialPropertyId) setSelectedProperty(initialPropertyId);
    if (initialResidentId) setSelectedResident(initialResidentId);
  }, [initialPropertyId, initialResidentId]);

  // Load (and score) residents for the SELECTED property only — never the whole
  // portfolio. Clearing the selection clears the table.
  useEffect(() => {
    if (!selectedProperty) {
      setResidents(null);
      setRollup(null);
      return;
    }
    let alive = true;
    setListLoading(true);
    setError("");
    listPropertyResidents(selectedProperty)
      .then((r) => {
        if (!alive) return;
        setResidents(r.residents);
        setRollup(r.rollup);
      })
      .catch((e) => alive && setError(errText(e)))
      .finally(() => {
        if (alive) setListLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [selectedProperty]);

  // KPI tiles: the selected property's rollup, or the portfolio overall.
  // KPI tiles reflect the selected property's rollup (from the per-property fetch).
  const kpi = rollup;
  const selectedName =
    propOptions?.find((p) => p.property_id === selectedProperty)?.name ?? selectedProperty;

  // The selected property's residents, searched + band-filtered, then sorted.
  const filtered = useMemo<ResidentRow[]>(() => {
    const rows = residents ?? [];
    const q = query.trim().toLowerCase();
    const matched = rows.filter(
      (r) =>
        (bandFilter === "all" || r.late_band === bandFilter) &&
        (q === "" ||
          r.unit_id.toLowerCase().includes(q) ||
          r.resident_id.toLowerCase().includes(q)),
    );
    const dir = sortAsc ? 1 : -1;
    const val = (r: ResidentRow): number | string => {
      switch (sortKey) {
        case "unit": return r.unit_id;
        case "tenure": return r.tenure_months;
        case "arrears": return r.expected_arrears;
        case "churn": return r.churn_probability ?? -1;
        case "serious": return r.serious_probability;
        case "balance": return r.current_balance;
        default: return r.late_probability;
      }
    };
    return [...matched].sort((a, b) => {
      const av = val(a);
      const bv = val(b);
      if (typeof av === "string" && typeof bv === "string") return dir * av.localeCompare(bv);
      return dir * ((av as number) - (bv as number));
    });
  }, [residents, bandFilter, query, sortKey, sortAsc]);

  const total = filtered.length;
  const capped = showAll ? filtered : filtered.slice(0, ROW_CAP);

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setSortAsc((v) => !v);
    } else {
      setSortKey(key);
      setSortAsc(key === "unit"); // text asc, numbers desc on first click
    }
    setShowAll(false);
  }

  const SortHeader = ({ label, keyName }: { label: string; keyName: SortKey }) => {
    const active = sortKey === keyName;
    return (
      <th aria-sort={active ? (sortAsc ? "ascending" : "descending") : "none"}>
        <button
          type="button"
          className="linklike icon-line"
          style={{ color: active ? "var(--text)" : "inherit", fontWeight: "inherit" }}
          onClick={() => toggleSort(keyName)}
        >
          {label}
          {active && (sortAsc ? <ArrowUp size={12} aria-hidden /> : <ArrowDown size={12} aria-hidden />)}
        </button>
      </th>
    );
  };

  return (
    <div className="app">
      <header>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <h1 style={{ margin: 0 }}>Residents</h1>
          <button
            type="button"
            className="linklike icon-line"
            aria-label="About these estimates"
            aria-expanded={showInfo}
            title="About these estimates"
            onClick={() => setShowInfo((v) => !v)}
          >
            <Info size={16} aria-hidden />
          </button>
        </div>
        <p>
          A decision-support view of forward-looking risk across current residents — to focus
          proactive outreach and retention, not to decide.
        </p>
      </header>

      {showInfo && (
        <RiskDisclaimer variant="banner" text={RESIDENT_DISCLAIMER} onViewModelCard={card ? scrollToModelCard : undefined} />
      )}

      {loading && (
        <div className="card">
          <p className="muted" style={{ margin: 0 }}>Scoring residents…</p>
        </div>
      )}

      {!loading && error && (
        <div className="card">
          <div className="error">{error}</div>
          <button className="btn-small btn-ghost" style={{ marginTop: 12 }} onClick={load}>
            Retry
          </button>
        </div>
      )}

      {!loading && !error && propOptions && (
        <>
          {/* Property selector */}
          <div className="res-prop-strip" style={{ marginTop: 8 }}>
            {propOptions.map((p) => {
              const on = selectedProperty === p.property_id;
              return (
                <button
                  type="button"
                  key={p.property_id}
                  className={`res-prop-card${on ? " active" : ""}`}
                  aria-pressed={on}
                  onClick={() => { setSelectedProperty(p.property_id); setSelectedResident(null); setShowAll(false); }}
                >
                  <span className="res-prop-name">{p.name}</span>
                  <span className="res-prop-meta">
                    {p.resident_count} resident{p.resident_count === 1 ? "" : "s"}
                  </span>
                </button>
              );
            })}
          </div>

          {/* KPI tiles */}
          {kpi && (
            <div className="stat-grid" style={{ marginTop: 18 }}>
              <div className="stat-tile">
                <div className="label">Predicted late-rate</div>
                <div className="value">{pct(kpi.predicted_late_rate)}</div>
                <div className="sub">any late payment next quarter</div>
              </div>
              <div className="stat-tile">
                <div className="label">Expected arrears</div>
                <div className="value">{usd(kpi.total_expected_arrears)}</div>
                <div className="sub">total projected next quarter</div>
              </div>
              <div className="stat-tile">
                <div className="label">Churn risk</div>
                <div className="value warn">{kpi.churn_risk_count}</div>
                <div className="sub">leases at renewal risk</div>
              </div>
              <div className="stat-tile">
                <div className="label">Serious flags</div>
                <div className="value bad">{kpi.serious_flag_count}</div>
                <div className="sub">routed to human review</div>
              </div>
            </div>
          )}

          {!selectedProperty && (
            <div className="card">
              <p className="muted" style={{ margin: 0 }}>
                Select a property above to view its residents and their forward-looking risk.
              </p>
            </div>
          )}

          {selectedProperty && listLoading && (
            <div className="card">
              <p className="muted" style={{ margin: 0 }}>Scoring residents at {selectedName}…</p>
            </div>
          )}

          {selectedProperty && !listLoading && residents && (
            <>
          {/* Resident table */}
          <div className="card">
            <h2>Residents — {selectedName}</h2>

            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 10, marginTop: 10 }}>
              <label className="field" style={{ flex: "1 1 200px", minWidth: 160 }}>
                <span className="sr-only">Search by unit or resident id</span>
                <span style={{ position: "relative", display: "flex" }}>
                  <Search
                    size={14}
                    aria-hidden
                    style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "var(--muted)", pointerEvents: "none" }}
                  />
                  <input
                    type="search"
                    value={query}
                    placeholder="Search unit or resident…"
                    aria-label="Search residents by unit or id"
                    style={{ paddingLeft: 34, width: "100%" }}
                    onChange={(e) => { setQuery(e.target.value); setShowAll(false); }}
                  />
                </span>
              </label>

              <div className="chip-row" style={{ marginTop: 0 }}>
                {BAND_FILTERS.map((f) => {
                  const on = bandFilter === f.key;
                  return (
                    <button
                      type="button"
                      key={f.key}
                      className="chip"
                      aria-pressed={on}
                      style={on ? { borderColor: "var(--accent)", color: "var(--accent-text)", background: "var(--accent-soft)" } : undefined}
                      onClick={() => { setBandFilter(f.key); setShowAll(false); }}
                    >
                      {f.label}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="table-scroll" style={{ marginTop: 8 }}>
              <table className="table rows-clickable">
                <thead>
                  <tr>
                    <SortHeader label="Unit" keyName="unit" />
                    <SortHeader label="Tenure" keyName="tenure" />
                    <SortHeader label="Late next Q" keyName="late" />
                    <th>Band</th>
                    <SortHeader label="Exp. balance" keyName="arrears" />
                    <SortHeader label="Churn" keyName="churn" />
                    <SortHeader label="Serious" keyName="serious" />
                    <SortHeader label="Balance" keyName="balance" />
                    <th>Top driver</th>
                  </tr>
                </thead>
                <tbody>
                  {capped.map((r) => (
                    <tr
                      key={r.resident_id}
                      onClick={() => setSelectedResident(r.resident_id)}
                      style={selectedResident === r.resident_id ? { background: "var(--panel2)" } : undefined}
                    >
                      <td>{r.unit_id}</td>
                      <td>{r.tenure_months} mo</td>
                      <td>{pct(r.late_probability)}</td>
                      <td>
                        <span className={`badge tone-${BAND_TONE[r.late_band]}`}>{BAND_LABEL[r.late_band]}</span>
                      </td>
                      <td>{usd(r.expected_arrears)}</td>
                      <td>{r.churn_probability == null ? <span className="muted">n/a</span> : pct(r.churn_probability)}</td>
                      <td>
                        <span className={`badge tone-${r.serious_band === "high" ? "bad" : churnTone(r.serious_band)}`}>
                          {r.serious_band === "high" ? "Flagged" : BAND_LABEL[r.serious_band]}
                        </span>
                      </td>
                      <td className={r.current_balance > 0 ? undefined : "secondary"}>{usd(r.current_balance)}</td>
                      <td className="secondary">{r.top_driver}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {total === 0 && (
              <p className="muted" style={{ margin: "10px 0 0", fontSize: 13 }}>
                No residents match this search or band.
              </p>
            )}

            {total > ROW_CAP && (
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10 }}>
                <span className="muted" style={{ fontSize: 12 }}>Showing {capped.length} of {total}</span>
                <button type="button" className="btn-small btn-ghost" onClick={() => setShowAll((v) => !v)}>
                  {showAll ? "Show top 25" : "Show all"}
                </button>
              </div>
            )}
          </div>

          {/* Detail region */}
          {selectedResident && (
            <div style={{ marginTop: 18 }}>
              <ResidentDetail
                key={selectedResident}
                residentId={selectedResident}
                onViewModelCard={card ? scrollToModelCard : undefined}
              />
            </div>
          )}
            </>
          )}
        </>
      )}

      {card && (
        <div ref={modelCardRef}>
          <ResidentModelCard card={card} />
        </div>
      )}
    </div>
  );
}
