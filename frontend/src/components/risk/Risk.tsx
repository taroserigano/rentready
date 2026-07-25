import { memo, useCallback, useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Cpu, Info, Link2, Search, X } from "lucide-react";
import { getRiskModelCard, listRisk } from "../../api";
import type { RiskBand, RiskListResponse, RiskModelCard as RiskModelCardT, RiskRow } from "../../types";
import { RiskCard } from "./RiskCard";
import { RiskDistribution } from "./RiskDistribution";
import { RiskModelCard } from "./RiskModelCard";
import { RiskWhatIf } from "./RiskWhatIf";
import { RiskChat } from "./RiskChat";
import { Badge } from "../Badge";
import { TechBadge } from "../TechBadge";
import { BAND_LABEL, BAND_TONE } from "./riskTone";

/** "xgboost" -> "XGBoost", "heuristic" -> "Heuristic". */
function techLabel(modelType?: string): string {
  if (!modelType) return "Heuristic";
  if (modelType.toLowerCase() === "xgboost") return "XGBoost";
  if (modelType.toLowerCase() === "histgb") return "HistGradientBoosting";
  return modelType.charAt(0).toUpperCase() + modelType.slice(1);
}

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

/** Module-level (not redefined per render) so React can reconcile instead of
 * remounting the header cells on every keystroke/filter/sort change. */
const SortHeader = memo(function SortHeader({
  label,
  keyName,
  sortKey,
  sortAsc,
  onToggle,
  hint,
}: {
  label: string;
  keyName: SortKey;
  sortKey: SortKey;
  sortAsc: boolean;
  onToggle: (key: SortKey) => void;
  /** Quick one-line description shown as a hover tooltip. */
  hint?: string;
}) {
  const active = sortKey === keyName;
  return (
    <th aria-sort={active ? (sortAsc ? "ascending" : "descending") : "none"} title={hint}>
      <button
        type="button"
        className="linklike icon-line"
        style={{ color: active ? "var(--text)" : "inherit", fontWeight: "inherit" }}
        onClick={() => onToggle(keyName)}
        title={hint}
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
});

/**
 * Resident Late-Payment Risk page. Ranked, decision-support view over every
 * saved applicant — a mandatory disclaimer banner, portfolio tiles, a band
 * distribution, a clickable ranked table, and a detail RiskCard.
 */
export function Risk({
  initialApplicantId,
  health,
}: {
  initialApplicantId?: string;
  health?: Record<string, unknown> | null;
}) {
  const [list, setList] = useState<RiskListResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(
    initialApplicantId ?? null,
  );

  // Client-side table controls (no backend change).
  const [query, setQuery] = useState("");
  const [bandFilter, setBandFilter] = useState<BandFilter>("all");
  const [sortKey, setSortKey] = useState<SortKey>("risk");
  const [sortAsc, setSortAsc] = useState(false);
  const [showAll, setShowAll] = useState(false);

  // Model-card governance modal (what's actually scoring this page).
  const [card, setCard] = useState<RiskModelCardT | null>(null);
  const [showModelCard, setShowModelCard] = useState(false);

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
  }, []);

  useEffect(load, [load]);

  useEffect(() => {
    getRiskModelCard().then(setCard).catch(() => {});
  }, []);

  // Close the model-card modal on Escape.
  useEffect(() => {
    if (!showModelCard) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setShowModelCard(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [showModelCard]);

  // Honour a deep-link that arrives after the first load.
  useEffect(() => {
    if (initialApplicantId) setSelectedId(initialApplicantId);
  }, [initialApplicantId]);

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
  // useCallback (keyed only on sortKey) so its identity is stable across
  // unrelated re-renders (typing in search, toggling a band chip), which lets
  // SortHeader's memo actually skip work instead of re-rendering every time.
  const toggleSort = useCallback(
    (key: SortKey) => {
      if (key === sortKey) {
        setSortAsc((v) => !v);
      } else {
        setSortKey(key);
        setSortAsc(key === "name");
      }
      setShowAll(false);
    },
    [sortKey],
  );

  return (
    <div className="app">
      <header>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <h1 style={{ margin: 0 }}>Late-Payment Risk</h1>
          {card && (
            <span
              className={`badge icon-line tone-${card.source === "model" ? "info" : "warn"}`}
              title={
                card.source === "model"
                  ? "Scored by a trained model — click the info icon for details"
                  : "No trained model loaded — falling back to the transparent heuristic"
              }
            >
              <Cpu size={12} aria-hidden />
              {techLabel(card.model_type)}
            </span>
          )}
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
          <TechBadge
            icon={Link2}
            label="LangChain"
            title="The chat rail's explanations are synthesized through a LangChain-wrapped Claude client."
          />
          <Badge on={!!health?.anthropic_key_set} label="Claude" tone="violet" />
        </div>
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
            No applicants to score yet. Use the Apply tab to upload a PDF or fill
            in details.
          </p>
        </div>
      )}

      {!loading && !error && list && list.rows.length > 0 && (
        <>
          <div className="stat-grid" style={{ marginTop: 8 }}>
            <div className="stat-tile">
              <div className="label">Average Risk</div>
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

          {/* Ranked Applicants + Late-Payment Risk come FIRST (right after the
              stat tiles, which never change size) so nothing above them can
              shift their position when an applicant is clicked. The chat rail
              below (whose height genuinely varies -- longer names wrap the
              starter chips differently, "Now asking about X" adds a line to
              the thread) is placed LAST specifically so its height changes
              have nothing below them left to push around. */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(320px, 1.1fr) minmax(340px, 1fr)",
              gap: 18,
              marginTop: 18,
              alignItems: "start",
            }}
          >
            <div className="card" style={{ marginTop: 0 }}>
              <h2>Ranked Applicants</h2>

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
                      <SortHeader
                        label="Applicant"
                        keyName="name"
                        sortKey={sortKey}
                        sortAsc={sortAsc}
                        onToggle={toggleSort}
                        hint="Applicant name"
                      />
                      <SortHeader
                        label="Risk"
                        keyName="risk"
                        sortKey={sortKey}
                        sortAsc={sortAsc}
                        onToggle={toggleSort}
                        hint="Predicted probability this applicant pays rent late (decision-support estimate, not a decision)"
                      />
                      <th title="Risk band for the estimate: Low, Moderate, or Elevated">Band</th>
                      <th title="The biggest factor behind this applicant's risk estimate">Top Driver</th>
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

            {selectedId ? (
              <div className="risk-detail-col" style={{ display: "grid", gap: 0 }}>
                <RiskCard key={selectedId} applicantId={selectedId} />
                <RiskWhatIf
                  key={`whatif-${selectedId}`}
                  applicantId={selectedId}
                />
              </div>
            ) : (
              <div className="card" style={{ marginTop: 0 }}>
                <h2>Late-Payment Risk</h2>
                <p className="muted" style={{ margin: 0 }}>
                  Select an applicant to see their risk detail.
                </p>
              </div>
            )}
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(260px, 1fr) minmax(320px, 1.4fr)",
              gap: 18,
              marginTop: 18,
              alignItems: "start",
            }}
          >
            <div className="card" style={{ marginTop: 0 }}>
              <h2>Band Distribution</h2>
              <RiskDistribution rows={list.rows} />
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
        </>
      )}

      {showModelCard && card && (
        <div
          className="modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Late-Payment Risk model card and metrics"
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
            <RiskModelCard card={card} />
          </div>
        </div>
      )}
    </div>
  );
}
