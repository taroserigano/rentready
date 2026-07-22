import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp, Search } from "lucide-react";
import { getRiskModelCard, listRisk } from "../../api";
import type { RiskBand, RiskListResponse, RiskModelCard, RiskRow } from "../../types";
import { RiskCard } from "./RiskCard";
import { RiskDistribution } from "./RiskDistribution";
import { ModelCard } from "./ModelCard";
import { RiskWhatIf } from "./RiskWhatIf";
import { RiskChat } from "./RiskChat";
import { BAND_LABEL, BAND_TONE } from "./riskTone";

/** Default number of rows shown before the "Show all" toggle. */
const ROW_CAP = 25;

type SortKey = "risk" | "name";
type BandFilter = "all" | RiskBand;

const BAND_FILTERS: { key: BandFilter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "low", label: BAND_LABEL.low },
  { key: "medium", label: BAND_LABEL.medium },
  { key: "high", label: BAND_LABEL.high },
];

function errText(e: unknown): string {
  if (e instanceof TypeError) {
    return "Could not reach the server. Is the backend running?";
  }
  if (e instanceof Error && e.message) return e.message;
  return "Could not reach the server. Is the backend running?";
}

/**
 * Resident Late-Payment Risk page. Ranked, decision-support view over every
 * saved applicant — a mandatory disclaimer banner, portfolio tiles, a band
 * distribution, a clickable ranked table, a detail RiskCard, and the model card.
 */
export function Risk({ initialApplicantId }: { initialApplicantId?: string }) {
  const [list, setList] = useState<RiskListResponse | null>(null);
  const [card, setCard] = useState<RiskModelCard | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(
    initialApplicantId ?? null,
  );
  const modelCardRef = useRef<HTMLDivElement>(null);

  // Client-side table controls (no backend change).
  const [query, setQuery] = useState("");
  const [bandFilter, setBandFilter] = useState<BandFilter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("risk");
  const [sortAsc, setSortAsc] = useState(false);
  const [showAll, setShowAll] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    setError("");
    listRisk()
      .then((r) => {
        setList(r);
        setSelectedId((cur) => cur ?? r.rows[0]?.applicant_id ?? null);
      })
      .catch((e) => setError(errText(e)))
      .finally(() => setLoading(false));
    // Model card is best-effort; the page works without it.
    getRiskModelCard()
      .then(setCard)
      .catch(() => {});
  }, []);

  useEffect(load, [load]);

  // Honour a deep-link that arrives after the first load.
  useEffect(() => {
    if (initialApplicantId) setSelectedId(initialApplicantId);
  }, [initialApplicantId]);

  const scrollToModelCard = () =>
    modelCardRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });

  // Filter (search + band) then sort. Fully client-side over the loaded rows.
  const filtered = useMemo<RiskRow[]>(() => {
    const rows = list?.rows ?? [];
    const q = query.trim().toLowerCase();
    const matched = rows.filter(
      (r) =>
        (bandFilter === "all" || r.band === bandFilter) &&
        (q === "" || r.name.toLowerCase().includes(q)),
    );
    const dir = sortAsc ? 1 : -1;
    return [...matched].sort((a, b) => {
      if (sortKey === "name") return dir * a.name.localeCompare(b.name);
      return dir * (a.probability - b.probability);
    });
  }, [list, query, bandFilter, sortKey, sortAsc]);

  const total = filtered.length;
  const capped = showAll ? filtered : filtered.slice(0, ROW_CAP);

  // The selected applicant's row (name + band seed the chat rail). Looked up
  // over ALL rows, not just the filtered view, so a filter can't blank it.
  const selectedRow = useMemo(
    () => list?.rows.find((r) => r.applicant_id === selectedId) ?? null,
    [list, selectedId],
  );

  // Toggle a column's sort; Risk defaults to desc, Name to asc on first click.
  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setSortAsc((v) => !v);
    } else {
      setSortKey(key);
      setSortAsc(key === "name");
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
          {active &&
            (sortAsc ? (
              <ArrowUp size={12} aria-hidden />
            ) : (
              <ArrowDown size={12} aria-hidden />
            ))}
        </button>
      </th>
    );
  };

  return (
    <div className="app">
      <header>
        <h1>Late-payment risk</h1>
        <p>
          A ranked, decision-support view of estimated late-payment probability
          across saved applicants — to focus a human review, not to decide.
        </p>
      </header>

      {loading && (
        <div className="card">
          <p className="muted" style={{ margin: 0 }}>
            Scoring applicants…
          </p>
        </div>
      )}

      {!loading && error && (
        <div className="card">
          <div className="error">{error}</div>
          <button
            className="btn-small btn-ghost"
            style={{ marginTop: 12 }}
            onClick={load}
          >
            Retry
          </button>
        </div>
      )}

      {!loading && !error && list && list.rows.length === 0 && (
        <div className="card">
          <p className="muted" style={{ margin: 0 }}>
            No applicants to score yet. Upload a PDF in Workspace or use the Apply
            tab.
          </p>
        </div>
      )}

      {!loading && !error && list && list.rows.length > 0 && (
        <>
          <div className="stat-grid" style={{ marginTop: 8 }}>
            <div className="stat-tile">
              <div className="label">Average risk</div>
              <div className="value">
                {Math.round(list.avg_probability * 100)}%
              </div>
              <div className="sub">calibrated mean across applicants</div>
            </div>
            <div className="stat-tile">
              <div className="label">Elevated</div>
              <div className="value bad">
                {Math.round(list.high_risk_pct * 100)}%
              </div>
              <div className="sub">routed to human review</div>
            </div>
            <div className="stat-tile">
              <div className="label">Scored</div>
              <div className="value">
                {list.scored}
                <span className="muted" style={{ fontSize: 16, fontWeight: 500 }}>
                  /{list.total}
                </span>
              </div>
              <div className="sub">applicants estimated</div>
            </div>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(280px, 1fr) 1.4fr",
              gap: 18,
              marginTop: 18,
              alignItems: "start",
            }}
          >
            <div className="card" style={{ marginTop: 0 }}>
              <h2>Band distribution</h2>
              <RiskDistribution rows={list.rows} />
            </div>

            <div className="card" style={{ marginTop: 0 }}>
              <h2>Ranked applicants</h2>

              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  alignItems: "center",
                  gap: 10,
                  marginTop: 10,
                }}
              >
                <label
                  className="field"
                  style={{ flex: "1 1 200px", minWidth: 160 }}
                >
                  <span className="sr-only">Search by name</span>
                  <span style={{ position: "relative", display: "flex" }}>
                    <Search
                      size={14}
                      aria-hidden
                      style={{
                        position: "absolute",
                        left: 12,
                        top: "50%",
                        transform: "translateY(-50%)",
                        color: "var(--muted)",
                        pointerEvents: "none",
                      }}
                    />
                    <input
                      type="search"
                      value={query}
                      placeholder="Search applicants…"
                      aria-label="Search applicants by name"
                      style={{ paddingLeft: 34, width: "100%" }}
                      onChange={(e) => {
                        setQuery(e.target.value);
                        setShowAll(false);
                      }}
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
                        style={
                          on
                            ? {
                                borderColor: "var(--accent)",
                                color: "var(--accent-text)",
                                background: "var(--accent-soft)",
                              }
                            : undefined
                        }
                        onClick={() => {
                          setBandFilter(f.key);
                          setShowAll(false);
                        }}
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
                      <SortHeader label="Applicant" keyName="name" />
                      <SortHeader label="Risk" keyName="risk" />
                      <th>Band</th>
                      <th>Top driver</th>
                    </tr>
                  </thead>
                  <tbody>
                    {capped.map((r) => (
                      <tr
                        key={r.applicant_id}
                        onClick={() => setSelectedId(r.applicant_id)}
                        style={
                          selectedId === r.applicant_id
                            ? { background: "var(--panel2)" }
                            : undefined
                        }
                      >
                        <td>{r.name}</td>
                        <td style={{ fontVariantNumeric: "tabular-nums" }}>
                          {Math.round(r.probability * 100)}%
                        </td>
                        <td>
                          <span className={`badge tone-${BAND_TONE[r.band]}`}>
                            {BAND_LABEL[r.band]}
                          </span>
                        </td>
                        <td className="secondary">{r.top_driver}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {total === 0 && (
                <p className="muted" style={{ margin: "10px 0 0", fontSize: 13 }}>
                  No applicants match this search or band.
                </p>
              )}

              {total > ROW_CAP && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    marginTop: 10,
                  }}
                >
                  <span className="muted" style={{ fontSize: 12 }}>
                    Showing {capped.length} of {total}
                  </span>
                  <button
                    type="button"
                    className="btn-small btn-ghost"
                    onClick={() => setShowAll((v) => !v)}
                  >
                    {showAll ? "Show top 25" : "Show all"}
                  </button>
                </div>
              )}
            </div>
          </div>

          {selectedId ? (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "minmax(0, 1.4fr) minmax(300px, 1fr)",
                gap: 18,
                marginTop: 18,
                alignItems: "start",
              }}
            >
              <div style={{ display: "grid", gap: 0 }}>
                <RiskCard
                  key={selectedId}
                  applicantId={selectedId}
                  onViewModelCard={card ? scrollToModelCard : undefined}
                />
                <RiskWhatIf
                  key={`whatif-${selectedId}`}
                  applicantId={selectedId}
                />
              </div>
              <div className="risk-chat-rail">
                <RiskChat
                  applicantId={selectedId}
                  applicantName={selectedRow?.name}
                  band={selectedRow?.band}
                  onSelectApplicant={setSelectedId}
                />
              </div>
            </div>
          ) : (
            <div style={{ marginTop: 18 }}>
              <RiskChat
                applicantId={null}
                onSelectApplicant={setSelectedId}
              />
            </div>
          )}
        </>
      )}

      {card && (
        <div ref={modelCardRef}>
          <ModelCard card={card} />
        </div>
      )}
    </div>
  );
}
