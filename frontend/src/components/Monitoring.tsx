import { useEffect, useState } from "react";
import { AlertTriangle } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  getEvalHistory,
  getMonitoring,
  pushDatadog,
  runEvals,
  type EvalSnapshot,
} from "../api";

// Series mapped to the categorical chart palette in a fixed order.
const SERIES = [
  { key: "eligibility_accuracy", name: "Eligibility", color: "var(--chart-1)" },
  { key: "extraction_field_accuracy", name: "Extraction", color: "var(--chart-2)" },
  { key: "ndcg_at_5", name: "NDCG@5", color: "var(--chart-3)" },
  { key: "judge_groundedness_pct", name: "Judge groundedness", color: "var(--chart-4)" },
  { key: "ragas_faithfulness", name: "RAGAS faithfulness", color: "var(--chart-5)" },
] as const;

const AXIS_TICK = { fill: "var(--chart-axis)", fontSize: 11 };

const TOOLTIP_STYLE = {
  background: "var(--panel-2)",
  border: "1px solid var(--line-strong)",
  borderRadius: 8,
  fontSize: 12,
  color: "var(--text)",
} as const;

export function Monitoring() {
  const [history, setHistory] = useState<EvalSnapshot[]>([]);
  const [live, setLive] = useState<Record<string, any> | null>(null);
  const [busy, setBusy] = useState(false);

  function load() {
    getEvalHistory(30)
      .then(setHistory)
      .catch(() => setHistory([]));
    getMonitoring()
      .then(setLive)
      .catch(() => setLive(null));
  }
  useEffect(load, []);

  async function runAndRefresh() {
    setBusy(true);
    try {
      await runEvals();
      load();
    } finally {
      setBusy(false);
    }
  }

  const data = history.map((h, i) => ({
    label: h.generated_at ? new Date(h.generated_at).toLocaleString() : `#${i + 1}`,
    short: `#${i + 1}`,
    ...Object.fromEntries(
      SERIES.map((s) => [
        s.key,
        h[s.key] == null ? null : Math.round((h[s.key] as number) * 100),
      ]),
    ),
  }));

  return (
    <div className="app">
      <header>
        <h1>Monitoring</h1>
        <p>
          Every evaluation run is persisted as a snapshot. This is the quality
          trend over time — the same metrics that gate CI, plus the live
          LLM-as-judge and RAGAS scores when a key is configured.
        </p>
      </header>

      {live && <LiveProduction live={live} onRefresh={load} />}

      <div className="card">
        <h2>Offline eval trend</h2>
        <button onClick={runAndRefresh} disabled={busy}>
          {busy ? "Running full suite…" : "Run all evals + record snapshot"}
        </button>
        <p className="muted" style={{ marginTop: 10 }}>
          {history.length} snapshot{history.length === 1 ? "" : "s"} recorded.
        </p>
      </div>

      {data.length > 0 ? (
        <div className="card">
          <h2>Offline quality over time (%)</h2>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: -16 }}>
              <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
              <XAxis dataKey="short" tick={AXIS_TICK} axisLine={false} tickLine={false} />
              <YAxis
                domain={[0, 100]}
                tick={AXIS_TICK}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={TOOLTIP_STYLE}
                labelFormatter={(_, p) => (p?.[0]?.payload?.label ?? "")}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: "var(--text-2)" }} />
              {SERIES.map((s) => (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  name={s.name}
                  stroke={s.color}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 3 }}
                  connectNulls
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="card">
          <p className="muted">
            No snapshots yet — click “Run all evals + record snapshot”.
          </p>
        </div>
      )}
    </div>
  );
}

function LiveProduction({
  live,
  onRefresh,
}: {
  live: Record<string, any>;
  onRefresh: () => void;
}) {
  const ov = live.overview ?? {};
  const lat = ov.latency ?? {};
  const fb = ov.feedback ?? {};
  const alerts: any[] = live.alerts ?? [];
  const drift = live.drift ?? {};
  const [dd, setDd] = useState<string>("");

  async function push() {
    setDd("pushing");
    try {
      const r = await pushDatadog();
      setDd(r.skipped ? `Datadog: ${r.reason}` : `Datadog: sent ${r.metrics} metrics (HTTP ${r.status}).`);
    } finally {
      onRefresh();
    }
  }

  return (
    <div className="card">
      <div className="toolbar">
        <h2 style={{ margin: 0 }}>Live production</h2>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn-small btn-ghost" onClick={onRefresh}>
            Refresh
          </button>
          <button
            className="btn-small btn-ghost"
            onClick={push}
            title={
              live.datadog_configured
                ? "Forward metrics to Datadog"
                : "Set DATADOG_API_KEY to enable"
            }
          >
            Push to Datadog
          </button>
        </div>
      </div>
      <p className="muted" style={{ marginTop: 4 }}>
        Telemetry from real requests served by the app (not the golden set).
        Use the Workspace to generate traffic, then refresh.
      </p>

      {alerts.length > 0 && (
        <div style={{ display: "grid", gap: 6, margin: "0 0 12px" }}>
          {alerts.map((a, i) => (
            <div key={i} className={`alert ${a.level === "critical" ? "bad" : "warn"}`}>
              <AlertTriangle size={15} style={{ flexShrink: 0 }} />
              <span>
                <b style={{ textTransform: "uppercase", fontSize: 11, letterSpacing: "0.06em" }}>
                  {a.level}
                </b>{" "}
                · {a.message}
              </span>
            </div>
          ))}
        </div>
      )}
      {alerts.length === 0 && (
        <p style={{ margin: "0 0 12px" }}>
          <span className="badge tone-info">
            <span className="dot good" />
            No active alerts
          </span>
        </p>
      )}

      <div className="stat-grid">
        <Stat label="Requests" value={ov.total_requests ?? 0} />
        <Stat label="Latency p50" value={`${Math.round(lat.p50_ms ?? 0)}ms`} />
        <Stat
          label="Latency p95"
          value={`${Math.round(lat.p95_ms ?? 0)}ms`}
          tone={(lat.p95_ms ?? 0) > 3000 ? "bad" : undefined}
        />
        <Stat
          label="Avg hallucinations"
          value={ov.mean_violations ?? 0}
          tone={(ov.mean_violations ?? 0) > 0 ? "bad" : undefined}
        />
        <Stat
          label="Thumbs up/down"
          value={`${fb.up ?? 0} / ${fb.down ?? 0}`}
        />
        <Stat
          label="Satisfaction"
          value={fb.satisfaction == null ? "—" : `${Math.round(fb.satisfaction * 100)}%`}
        />
      </div>

      {drift.enough_data && (
        <p className="muted" style={{ fontSize: 12, marginTop: 12 }}>
          Drift (recent vs previous window): latency{" "}
          {(drift.latency.pct_change * 100).toFixed(0)}%
          {drift.latency.drifted ? " ⚠" : ""} · violations{" "}
          {drift.violations.abs_change >= 0 ? "+" : ""}
          {drift.violations.abs_change}
          {drift.violations.drifted ? " ⚠" : ""}
        </p>
      )}
      {dd && (
        <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
          {dd}
        </p>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: any;
  tone?: "good" | "warn" | "bad";
}) {
  return (
    <div className="stat-tile">
      <div className="label">{label}</div>
      <div className={`value${tone ? ` ${tone}` : ""}`}>{value}</div>
    </div>
  );
}
