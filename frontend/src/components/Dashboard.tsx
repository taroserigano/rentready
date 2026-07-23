import { useCallback, useEffect, useState, type ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { DashboardStats } from "../types";
import { getDashboardStats } from "../api";
import { Avatar } from "./Avatar";

type Verdict = "qualified" | "needs_review" | "not_qualified";

const VERDICT_LABELS: Record<Verdict, string> = {
  qualified: "Qualified",
  needs_review: "Needs review",
  not_qualified: "Not qualified",
};

const VERDICT_COLORS: Record<Verdict, string> = {
  qualified: "var(--chart-2)",
  needs_review: "var(--chart-3)",
  not_qualified: "var(--chart-4)",
};

const VERDICT_TONES: Record<Verdict, string> = {
  qualified: "tone-good",
  needs_review: "tone-warn",
  not_qualified: "tone-bad",
};

// Short category names keep the x-axis readable. Any endpoint added later
// without an entry here still degrades gracefully via _titleCase() below,
// rather than falling back to a raw snake_case name.
const ENDPOINT_LABELS: Record<string, string> = {
  eligibility: "Eligibility",
  recommend: "Match",
  ask: "Q&A",
  apply: "Apply",
  properties: "Browse",
  candidates: "Candidates",
  residents_get: "Resident",
  residents_list: "Residents",
  residents_by_property: "By property",
  residents_health: "Health",
  residents_portfolio_summary: "Portfolio",
  residents_properties: "Res. props",
  residents_score: "Rescore",
  residents_chat: "Res. chat",
  residents_chat_stream: "Res. chat (live)",
  residents_model_card: "Res. model",
  risk_get: "Risk",
  risk_list: "Risk list",
  risk_score: "Risk score",
  risk_chat: "Risk chat",
  risk_chat_stream: "Risk chat (live)",
  risk_model_card: "Risk model",
  concierge_ask: "Concierge",
  concierge_ask_stream: "Concierge (live)",
  tours_slots: "Tour slots",
  tours_staff: "Tour staff",
  tours_chat: "Tour chat",
  tours_list: "Tour list",
};

function _titleCase(endpoint: string): string {
  return endpoint.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

const AXIS_TICK = { fill: "var(--chart-axis)", fontSize: 11 };

const TOOLTIP_STYLE = {
  background: "var(--panel-2)",
  border: "1px solid var(--line-strong)",
  borderRadius: 8,
  fontSize: 12,
  color: "var(--text)",
} as const;

const TOOLTIP_CURSOR = { fill: "rgba(255,255,255,0.04)" };

function money(n: number): string {
  return `$${n.toLocaleString()}`;
}

function when(s: string): string {
  return new Date(s).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    setError("");
    getDashboardStats()
      .then(setStats)
      .catch((e) =>
        setError(
          e instanceof TypeError || !(e instanceof Error) || !e.message
            ? "Could not reach the server. Is the backend running?"
            : e.message,
        ),
      )
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  return (
    <div className="app">
      <header>
        <h1>Dashboard</h1>
        <p>The whole system at a glance: applicants, homes, traffic, feedback.</p>
      </header>

      {loading && (
        <div className="card">
          <p className="muted">Loading dashboard…</p>
        </div>
      )}

      {!loading && error && (
        <div className="card">
          <div className="error">{error}</div>
          <button className="btn-small btn-ghost" onClick={load} style={{ marginTop: 12 }}>
            Retry
          </button>
        </div>
      )}

      {!loading && !error && stats && <DashboardBody stats={stats} onRefresh={load} />}
    </div>
  );
}

/** How many neighborhood bars to show before rolling the rest into "+N more". */
const MAX_AREA_BARS = 8;

function DashboardBody({
  stats,
  onRefresh,
}: {
  stats: DashboardStats;
  onRefresh: () => void;
}) {
  const verdictData = (Object.keys(VERDICT_LABELS) as Verdict[]).map((v) => ({
    name: VERDICT_LABELS[v],
    count: stats.verdicts[v] ?? 0,
    color: VERDICT_COLORS[v],
  }));

  const requestsByEndpoint = stats.traffic.requests_by_endpoint ?? {};
  const requestVolumeData = Object.entries(requestsByEndpoint).map(([endpoint, count]) => ({
    name: ENDPOINT_LABELS[endpoint] ?? _titleCase(endpoint),
    count,
  }));

  const byArea = stats.properties.by_area ?? {};
  const areaEntries = Object.entries(byArea);
  const areaData = areaEntries.slice(0, MAX_AREA_BARS).map(([name, count]) => ({ name, count }));
  const areasOmitted = Math.max(0, areaEntries.length - MAX_AREA_BARS);

  const petsPct =
    stats.properties.total > 0
      ? Math.round((stats.properties.pets_allowed / stats.properties.total) * 100)
      : 0;

  const feedbackTotal = stats.feedback.up + stats.feedback.down;
  const satisfactionPct =
    feedbackTotal > 0 ? Math.round((stats.feedback.up / feedbackTotal) * 100) : null;

  const showVerdictChart = stats.applicants_total > 0;
  const showRequestVolumeChart = stats.traffic.events_total > 0 && requestVolumeData.length > 0;
  const showAreaChart = areaData.length > 0;

  return (
    <>
      {stats.applicants_total === 0 ? (
        <div className="card" style={{ marginTop: 24 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <h2 style={{ margin: 0 }}>Getting started</h2>
            <button className="btn-small btn-ghost" onClick={onRefresh}>
              Refresh
            </button>
          </div>
          <p className="muted" style={{ marginTop: 10 }}>
            No applicants yet. Try the Apply tab to get started.
          </p>
        </div>
      ) : null}

      {/* ---------------- Applicants ---------------- */}
      <div className="eyebrow" style={{ marginTop: 24 }}>
        Applicants
      </div>
      <div className="stat-grid" style={{ marginTop: 10 }}>
        <Tile label="Total" value={stats.applicants_total} />
        <Tile label="Qualified" value={stats.verdicts.qualified} valueClass="good" />
        <Tile label="Needs review" value={stats.verdicts.needs_review} valueClass="warn" />
        <Tile label="Not qualified" value={stats.verdicts.not_qualified} valueClass="bad" />
      </div>

      {showVerdictChart && (
        <div className="card">
          <h2>Eligibility breakdown</h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart
              data={verdictData}
              layout="vertical"
              margin={{ top: 8, right: 16, bottom: 8, left: 16 }}
            >
              <CartesianGrid stroke="var(--chart-grid)" horizontal={false} />
              <XAxis
                type="number"
                allowDecimals={false}
                tick={AXIS_TICK}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={90}
                tick={AXIS_TICK}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip cursor={TOOLTIP_CURSOR} contentStyle={TOOLTIP_STYLE} />
              <Bar
                dataKey="count"
                name="Applicants"
                barSize={18}
                radius={[0, 4, 4, 0]}
                isAnimationActive={false}
              >
                {verdictData.map((d) => (
                  <Cell key={d.name} fill={d.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {stats.recent_applicants.length > 0 && (
        <div className="card">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <h2 style={{ margin: 0 }}>Recent applicants</h2>
            <button className="btn-small btn-ghost" onClick={onRefresh}>
              Refresh
            </button>
          </div>
          <table className="table" style={{ marginTop: 12 }}>
            <thead>
              <tr>
                <th>Name</th>
                <th>When</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {stats.recent_applicants.map((a) => (
                <tr key={a.id}>
                  <td>
                    <span className="cell-avatar">
                      <Avatar name={a.name} size={28} />
                      {a.name}
                    </span>
                  </td>
                  <td className="secondary">{when(a.created_at)}</td>
                  <td>
                    <span className={`badge ${VERDICT_TONES[a.verdict]}`}>
                      {VERDICT_LABELS[a.verdict]}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ---------------- Homes ---------------- */}
      <div className="eyebrow" style={{ marginTop: 32 }}>
        Homes
      </div>
      <div className="stat-grid" style={{ marginTop: 10 }}>
        <Tile
          label="Properties listed"
          value={stats.properties.total}
          sub={`Rent ${money(stats.properties.rent_min)}–${money(stats.properties.rent_max)}`}
        />
        <Tile
          label="Pet-friendly"
          value={stats.properties.pets_allowed}
          sub={`${petsPct}% of listings`}
        />
        <Tile label="Neighborhoods" value={stats.properties.areas} />
      </div>

      {showAreaChart && (
        <div className="card">
          <h2>Homes by neighborhood</h2>
          <ResponsiveContainer width="100%" height={Math.max(160, areaData.length * 34)}>
            <BarChart
              data={areaData}
              layout="vertical"
              margin={{ top: 8, right: 16, bottom: 8, left: 16 }}
            >
              <CartesianGrid stroke="var(--chart-grid)" horizontal={false} />
              <XAxis
                type="number"
                allowDecimals={false}
                tick={AXIS_TICK}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={120}
                tick={AXIS_TICK}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip cursor={TOOLTIP_CURSOR} contentStyle={TOOLTIP_STYLE} />
              <Bar
                dataKey="count"
                name="Homes"
                fill="var(--chart-5)"
                barSize={16}
                radius={[0, 4, 4, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>
          {areasOmitted > 0 && (
            <p className="muted" style={{ marginTop: 8, fontSize: "var(--fs-xs)" }}>
              +{areasOmitted} more neighborhood{areasOmitted > 1 ? "s" : ""}
            </p>
          )}
        </div>
      )}

      {/* ---------------- Traffic ---------------- */}
      <div className="eyebrow" style={{ marginTop: 32 }}>
        Traffic
      </div>
      <div className="stat-grid" style={{ marginTop: 10 }}>
        <Tile label="Requests logged" value={stats.traffic.events_total} sub="last 500 events" />
        <Tile
          label="Faithfulness flags"
          value={stats.traffic.faithfulness_violations}
          valueClass={stats.traffic.faithfulness_violations > 0 ? "warn" : undefined}
        />
        <Tile label="Outliers excluded" value={stats.traffic.outliers_excluded ?? 0} />
      </div>

      {showRequestVolumeChart && (
        <div className="card">
          <h2>Requests by endpoint</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={requestVolumeData}
              margin={{ top: 8, right: 16, bottom: 48, left: -8 }}
            >
              <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
              <XAxis
                dataKey="name"
                tick={AXIS_TICK}
                axisLine={false}
                tickLine={false}
                interval={0}
                angle={-40}
                textAnchor="end"
                height={70}
              />
              <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} allowDecimals={false} />
              <Tooltip cursor={TOOLTIP_CURSOR} contentStyle={TOOLTIP_STYLE} />
              <Bar
                dataKey="count"
                name="Requests"
                fill="var(--chart-1)"
                barSize={18}
                radius={[4, 4, 0, 0]}
                isAnimationActive={false}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ---------------- Feedback ---------------- */}
      <div className="eyebrow" style={{ marginTop: 32 }}>
        Feedback
      </div>
      <div className="stat-grid" style={{ marginTop: 10 }}>
        <Tile label="Thumbs up" value={stats.feedback.up} valueClass="good" />
        <Tile label="Thumbs down" value={stats.feedback.down} valueClass="bad" />
        <Tile
          label="Satisfaction"
          value={satisfactionPct === null ? "—" : `${satisfactionPct}%`}
          sub={feedbackTotal > 0 ? `${feedbackTotal} rated` : "no ratings yet"}
        />
      </div>

      {satisfactionPct !== null && (
        <div className="card">
          <div className="mini-row">
            <span className="k">Positive rate</span>
            <span className="v">{satisfactionPct}%</span>
          </div>
          <div className="meter" style={{ marginTop: 6 }}>
            <i
              className={satisfactionPct >= 70 ? "good" : satisfactionPct >= 40 ? "warn" : "bad"}
              style={{ width: `${satisfactionPct}%` }}
            />
          </div>
        </div>
      )}
    </>
  );
}

function Tile({
  label,
  value,
  sub,
  valueClass,
}: {
  label: string;
  value: ReactNode;
  sub?: string;
  valueClass?: "good" | "warn" | "bad";
}) {
  return (
    <div className="stat-tile">
      <div className="label">{label}</div>
      <div className={`value${valueClass ? ` ${valueClass}` : ""}`}>{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}
