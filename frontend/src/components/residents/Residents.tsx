import { memo, useCallback, useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Info, Search, X } from "lucide-react";
import { getResidentModelCard, getResidentProperties, listPropertyResidents } from "../../api";
import type {
  PropertyResidentRollup,
  ResidentModelCard as ResidentModelCardT,
  ResidentPropertyOption,
  ResidentRow,
  RiskBand,
} from "../../types";
import { ResidentDetail } from "./ResidentDetail";
import { ResidentModelCard } from "./ResidentModelCard";
import { PropertyHealthRanking } from "./PropertyHealthRanking";
import { ResidentsChat } from "./ResidentsChat";
import { BAND_LABEL, BAND_TONE, churnTone, pct, usd } from "./residentsTone";

const ROW_CAP = 25;

type SortKey = "name" | "unit" | "tenure" | "late" | "arrears" | "churn" | "serious" | "balance";
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

/** Module-level (not redefined per render) so React can reconcile instead of
 * remounting the header cells on every keystroke/filter/sort change. */
const SortHeader = memo(function SortHeader({
  label,
  keyName,
  sortKey,
  sortAsc,
  onToggle,
}: {
  label: string;
  keyName: SortKey;
  sortKey: SortKey;
  sortAsc: boolean;
  onToggle: (key: SortKey) => void;
}) {
  const active = sortKey === keyName;
  return (
    <th aria-sort={active ? (sortAsc ? "ascending" : "descending") : "none"}>
      <button
        type="button"
        className="linklike icon-line"
        style={{ color: active ? "var(--text)" : "inherit", fontWeight: "inherit" }}
        onClick={() => onToggle(keyName)}
      >
        {label}
        {active && (sortAsc ? <ArrowUp size={12} aria-hidden /> : <ArrowDown size={12} aria-hidden />)}
      </button>
    </th>
  );
});

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
  const [showModelCard, setShowModelCard] = useState(false);
  const [selectedResident, setSelectedResident] = useState<string | null>(initialResidentId ?? null);

  const [query, setQuery] = useState("");
  const [bandFilter, setBandFilter] = useState<BandFilter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("late");
  const [sortAsc, setSortAsc] = useState(false);

  // Close the model-card modal on Escape.
  useEffect(() => {
    if (!showModelCard) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setShowModelCard(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showModelCard]);

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
  const selectedResidentName = residents?.find(
    (r) => r.resident_id === selectedResident,
  )?.name;

  // Drill into a property from the health ranking or a chat result.
  const selectProperty = useCallback((id: string) => {
    setSelectedProperty(id);
    setSelectedResident(null);
  }, []);

  // The selected property's residents, searched + band-filtered, then sorted.
  const filtered = useMemo<ResidentRow[]>(() => {
    const rows = residents ?? [];
    const q = query.trim().toLowerCase();
    const matched = rows.filter(
      (r) =>
        (bandFilter === "all" || r.late_band === bandFilter) &&
        (q === "" ||
          r.unit_id.toLowerCase().includes(q) ||
          r.resident_id.toLowerCase().includes(q) ||
          r.name.toLowerCase().includes(q)),
    );
    const dir = sortAsc ? 1 : -1;
    const val = (r: ResidentRow): number | string => {
      switch (sortKey) {
        case "name": return r.name;
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

  // useCallback (keyed only on sortKey) so its identity is stable across
  // unrelated re-renders (typing in search, toggling a band chip), which lets
  // SortHeader's memo actually skip work instead of re-rendering every time.
  const toggleSort = useCallback(
    (key: SortKey) => {
      if (key === sortKey) {
        setSortAsc((v) => !v);
      } else {
        setSortKey(key);
        setSortAsc(key === "unit" || key === "name"); // text asc, numbers desc on first click
      }
    },
    [sortKey],
  );

  return (
    <div className="app">
      <header>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <h1 style={{ margin: 0 }}>Residents</h1>
          {card && (
            <button
              type="button"
              className="linklike icon-line"
              aria-label="Model card & how we measure"
              title="Model card & how we measure"
              onClick={() => setShowModelCard(true)}
            >
              <Info size={16} aria-hidden />
            </button>
          )}
        </div>
        <p>
          A decision-support view of forward-looking risk across current residents — to focus
          proactive outreach and retention, not to decide.
        </p>
      </header>

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
                  onClick={() => { setSelectedProperty(p.property_id); setSelectedResident(null); }}
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

          {/* No property selected → the regional-director health ranking
              (best→worst) in place of a bare prompt, plus a portfolio chat. */}
          {!selectedProperty && (
            <>
              <div style={{ marginTop: 18 }}>
                <PropertyHealthRanking onSelect={selectProperty} />
              </div>
              <div style={{ marginTop: 18 }}>
                <ResidentsChat
                  residentId={null}
                  propertyId={null}
                  onSelectProperty={selectProperty}
                />
              </div>
            </>
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
                    onChange={(e) => setQuery(e.target.value)}
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
                      onClick={() => setBandFilter(f.key)}
                    >
                      {f.label}
                    </button>
                  );
                })}
              </div>
            </div>

            <div
              className={`table-scroll${total >= ROW_CAP ? " table-scroll--capped" : ""}`}
              style={{ marginTop: 8 }}
            >
              <table className="table rows-clickable">
                <thead>
                  <tr>
                    <SortHeader label="Resident" keyName="name" sortKey={sortKey} sortAsc={sortAsc} onToggle={toggleSort} />
                    <SortHeader label="Unit" keyName="unit" sortKey={sortKey} sortAsc={sortAsc} onToggle={toggleSort} />
                    <SortHeader label="Tenure" keyName="tenure" sortKey={sortKey} sortAsc={sortAsc} onToggle={toggleSort} />
                    <SortHeader label="Late next Q" keyName="late" sortKey={sortKey} sortAsc={sortAsc} onToggle={toggleSort} />
                    <th>Band</th>
                    <SortHeader label="Exp. balance" keyName="arrears" sortKey={sortKey} sortAsc={sortAsc} onToggle={toggleSort} />
                    <SortHeader label="Churn" keyName="churn" sortKey={sortKey} sortAsc={sortAsc} onToggle={toggleSort} />
                    <SortHeader label="Serious" keyName="serious" sortKey={sortKey} sortAsc={sortAsc} onToggle={toggleSort} />
                    <SortHeader label="Balance" keyName="balance" sortKey={sortKey} sortAsc={sortAsc} onToggle={toggleSort} />
                    <th>Top driver</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((r) => (
                    <tr
                      key={r.resident_id}
                      onClick={() => setSelectedResident(r.resident_id)}
                      style={selectedResident === r.resident_id ? { background: "var(--panel2)" } : undefined}
                    >
                      <td style={{ fontWeight: 600 }}>{r.name}</td>
                      <td className="secondary">{r.unit_id}</td>
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

            {total >= ROW_CAP && (
              <p className="muted" style={{ margin: "10px 0 0", fontSize: 12 }}>
                {total} residents — scroll for more
              </p>
            )}
          </div>

          {/* Detail + chat rail. With a resident selected the detail and a
              sticky chat rail sit side by side; otherwise the chat runs full
              width below the table, scoped to the property. */}
          {selectedResident ? (
            <div className="residents-detail-layout" style={{ marginTop: 18 }}>
              <div>
                <ResidentDetail
                  key={selectedResident}
                  residentId={selectedResident}
                />
              </div>
              <aside className="risk-chat-rail">
                <ResidentsChat
                  residentId={selectedResident}
                  residentName={selectedResidentName}
                  propertyId={selectedProperty}
                  propertyName={selectedName}
                  onSelectProperty={selectProperty}
                />
              </aside>
            </div>
          ) : (
            <div style={{ marginTop: 18 }}>
              <ResidentsChat
                residentId={null}
                propertyId={selectedProperty}
                propertyName={selectedName}
                onSelectProperty={selectProperty}
              />
            </div>
          )}
            </>
          )}
        </>
      )}

      {showModelCard && card && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Resident risk model card and metrics"
          onClick={() => setShowModelCard(false)}
        >
          <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
            <button
              type="button"
              className="modal-close"
              aria-label="Close"
              onClick={() => setShowModelCard(false)}
            >
              <X size={18} aria-hidden />
            </button>
            <ResidentModelCard card={card} />
          </div>
        </div>
      )}
    </div>
  );
}
