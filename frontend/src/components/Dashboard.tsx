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

// Short category names keep the x-axis readable — no rotated labels.
const ENDPOINT_LABELS: Record<string, string> = {
  eligibility: "Eligibility",
  recommend: "Match",
  ask: "Q&A",
  apply: "Apply",
  properties: "Browse",
};

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

  const latencyData = Object.entries(stats.traffic.avg_latency_ms_by_endpoint).map(
    ([endpoint, ms]) => ({
      name: ENDPOINT_LABELS[endpoint] ?? endpoint,
      ms,
    }),
  );

  const showVerdictChart = stats.applicants_total > 0;
  const showLatencyChart = stats.traffic.events_total > 0 && latencyData.length > 0;

  return (
    <>
      <div className="stat-grid" style={{ marginTop: 24 }}>
        <Tile label="Applicants" value={stats.applicants_total} />
        <Tile label="Qualified" value={stats.verdicts.qualified} valueClass="good" />
        <Tile
          label="Properties listed"
          value={stats.properties.total}
          sub={`Rent ${money(stats.properties.rent_min)}–${money(stats.properties.rent_max)}`}
        />
        <Tile
          label="Feedback"
          value={
            <>
              <span style={{ color: "var(--good-text)" }}>{stats.feedback.up}↑</span>
              <span style={{ color: "var(--muted)" }}> · </span>
              <span style={{ color: "var(--bad-text)" }}>{stats.feedback.down}↓</span>
            </>
          }
        />
      </div>

      {stats.applicants_total === 0 ? (
        <div className="card">
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
            No applicants yet. Try the Apply tab or upload a PDF in Workspace.
          </p>
        </div>
      ) : (
        <>
          {(showVerdictChart || showLatencyChart) && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 18,
                marginTop: 18,
              }}
            >
              {showVerdictChart && (
                <div className="card" style={{ marginTop: 0 }}>
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

              {showLatencyChart && (
                <div className="card" style={{ marginTop: 0 }}>
                  <h2>Average response time (ms)</h2>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart
                      data={latencyData}
                      margin={{ top: 8, right: 16, bottom: 8, left: -8 }}
                    >
                      <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
                      <XAxis
                        dataKey="name"
                        tick={AXIS_TICK}
                        axisLine={false}
                        tickLine={false}
                        interval={0}
                      />
                      <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} />
                      <Tooltip cursor={TOOLTIP_CURSOR} contentStyle={TOOLTIP_STYLE} />
                      <Bar
                        dataKey="ms"
                        name="Avg ms"
                        fill="var(--chart-1)"
                        barSize={18}
                        radius={[4, 4, 0, 0]}
                        isAnimationActive={false}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}

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
        </>
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
